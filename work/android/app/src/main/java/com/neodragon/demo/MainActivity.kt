package com.neodragon.demo

import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.max
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val runner = QnnRunner(this)
        setContent { MaterialTheme(colorScheme = darkColorScheme()) { Ui(runner) } }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun Ui(runner: QnnRunner) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()

    var prompt by remember { mutableStateOf("a red fox walking through fresh snow, cinematic") }
    var seed by remember { mutableStateOf("0") }
    var log by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("") }
    var progress by remember { mutableFloatStateOf(0f) }

    var image by remember { mutableStateOf<Bitmap?>(null) }
    // The user's photo for image-to-video. Null = generate a first frame with SSD1B.
    var srcImage by remember { mutableStateOf<Bitmap?>(null) }
    var frames by remember { mutableStateOf<List<Bitmap>>(emptyList()) }
    var frameIdx by remember { mutableIntStateOf(0) }
    var playing by remember { mutableStateOf(true) }
    var fps by remember { mutableIntStateOf(12) }
    var showLog by remember { mutableStateOf(false) }
    var saving by remember { mutableStateOf(false) }
    var savedMsg by remember { mutableStateOf("") }

    // Models can be pushed while the app is running, so the count is re-read on demand;
    // keying it on the log meant it was computed once and then read 0/12 forever.
    var refresh by remember { mutableIntStateOf(0) }
    val missing = remember(refresh) { ModelStore.missing(runner) }

    // ---- model download --------------------------------------------------------
    val settings = remember { Settings(ctx) }
    var baseUrl by remember { mutableStateOf(settings.baseUrl) }
    var token by remember { mutableStateOf(settings.token) }
    var showModels by remember { mutableStateOf(false) }
    var downloading by remember { mutableStateOf(false) }
    var dlName by remember { mutableStateOf("") }
    var dlFileDone by remember { mutableLongStateOf(0L) }
    var dlFileTotal by remember { mutableLongStateOf(0L) }
    var dlBase by remember { mutableLongStateOf(0L) }   // bytes finished before dlName
    var dlTotal by remember { mutableLongStateOf(0L) }
    // Poll the service's process-wide state. The UI may not have existed when the
    // download started, so it attaches to whatever is already running rather than
    // owning the transfer.
    LaunchedEffect(Unit) {
        while (true) {
            val st = DownloadService.State
            downloading = st.running
            dlName = st.name
            dlBase = 0L
            dlFileDone = st.doneBytes
            dlFileTotal = if (st.totalBytes > 0) st.totalBytes else st.totalFileBytes
            dlTotal = st.totalBytes
            if (st.finished && st.message.isNotEmpty()) {
                if (savedMsg != st.message) { savedMsg = st.message; refresh++ }
            }
            delay(500)
        }
    }

    // PickVisualMedia is the modern picker: no READ_MEDIA_IMAGES permission needed, so
    // the app stays permission-free.
    // The picked photo BEFORE cropping. srcImage is only set once the user approves a
    // region, so an accidental pick never silently changes what will be animated.
    var cropCandidate by remember { mutableStateOf<Bitmap?>(null) }
    val picker = androidx.activity.compose.rememberLauncherForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            runCatching {
                ctx.contentResolver.openInputStream(uri).use { ins ->
                    // Downsample on decode: a 108 MP phone photo is ~400 MB as ARGB_8888
                    // and would OOM before it ever reached the crop dialog. 2048 on the
                    // long edge is far more than the 512x320 the model takes.
                    val opts = android.graphics.BitmapFactory.Options().apply {
                        inJustDecodeBounds = true
                    }
                    ctx.contentResolver.openInputStream(uri).use {
                        android.graphics.BitmapFactory.decodeStream(it, null, opts)
                    }
                    var ss = 1
                    while (max(opts.outWidth, opts.outHeight) / ss > 2048) ss *= 2
                    val o2 = android.graphics.BitmapFactory.Options().apply {
                        inSampleSize = ss
                    }
                    cropCandidate = android.graphics.BitmapFactory.decodeStream(ins, null, o2)
                }
            // `say` is a local fun declared further down, so it is not in scope here.
            }.onFailure { android.util.Log.w("neodragon", "could not read image", it) }
        }
    }

    cropCandidate?.let { cand ->
        CropDialog(
            source = cand,
            targetW = VideoStructure.cropWidth(ctx),
            targetH = VideoStructure.cropHeight(ctx),
            onCancel = { cropCandidate = null },
            onDone = { cropped -> srcImage = cropped; cropCandidate = null },
        )
    }

    // Mirrored to logcat as well as the on-screen panel: the panel is the only place the
    // per-stage timings show up otherwise, and reading them off a screenshot is not a
    // measurement. `adb logcat -s neodragon` gets them.
    fun say(s: String) {
        android.util.Log.i("neodragon", s)
        log = (log + s + "\n").takeLast(12000)
    }

    // ---- preload the expensive context binaries ---------------------------------
    // CLIP G is 1.4 GB and takes 1897 ms to map, against 40 ms of actual compute -- it
    // was 43% of the first frame before it was kept resident. That load is storage-bound
    // and cannot be made faster, but it CAN be moved off the critical path: the user
    // spends several seconds typing a prompt before pressing anything, so mapping it on a
    // background thread at start makes the first generation warm instead of cold
    // (7.5 s -> 2.2 s).
    //
    // Deliberately best-effort: if the model is absent (fresh install, nothing downloaded
    // yet) this must do nothing rather than throw, and it must never block the UI.
    var preloaded by remember { mutableStateOf("") }
    var warming by remember { mutableStateOf(false) }

    // Device support, decided by a 58 KB canary graph shipped in the APK rather than by
    // a SoC allowlist. Runs once, before the download is offered -- finding out the chip
    // is wrong AFTER pulling 8 GB is the outcome this exists to prevent.
    var devCheck by remember { mutableStateOf<DeviceCheck.Result?>(null) }
    var confirmDownload by remember { mutableStateOf(false) }
    val notifPerm = androidx.activity.compose.rememberLauncherForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.RequestPermission()
    ) { }
    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            val r = DeviceCheck.run(ctx, runner)
            devCheck = r
            say(when (r) {
                is DeviceCheck.Result.Ok ->
                    "device OK (${DeviceCheck.describeDevice()}), canary %.1f dB".format(r.snrDb)
                is DeviceCheck.Result.Unsupported ->
                    "UNSUPPORTED DEVICE (${DeviceCheck.describeDevice()}): " + r.detail
                is DeviceCheck.Result.Fp16Suspect ->
                    "fp16 canary FAILED (%.1f dB) on ${DeviceCheck.describeDevice()}".format(r.snrDb)
                is DeviceCheck.Result.Inconclusive ->
                    "device check inconclusive: " + r.detail
            })
        }
    }
    // Keyed on model availability, NOT on `refresh` -- run() bumps refresh when a
    // generation finishes, which re-fired this after every image and made the warming
    // indicator flicker. This way it runs at start and again if a download completes.
    LaunchedEffect(missing.isEmpty(), srcImage != null) {
        if (missing.isNotEmpty()) return@LaunchedEffect
        withContext(Dispatchers.IO) {
            warming = true
            // The whole first-frame path, biggest first. Preloading clipg alone still
            // left the FIRST generation at 5552 ms because ssd1bunet (1.3 GB) was cold;
            // with all four mapped up front the first run is as fast as the warm ones.
            // Order matters: clipg is the single most expensive map, so start it first.
            // Image-to-video never touches the SSD1B first-frame path, so mapping its
            // 1.68 GB would be pure waste -- and it is most of the first-tap wait.
            val need = if (srcImage != null) listOf("clipg")
                       else listOf("clipg", "ssd1bunet", "clipl", "ssd1bvaedec")
            for (n in need) {
                if (!runner.isReady(n)) continue
                runCatching {
                    val t = System.nanoTime()
                    runner.handleOf(n)
                    val ms = (System.nanoTime() - t) / 1e6
                    say("preloaded %s in %.0f ms".format(n, ms))
                    preloaded = n
                }
            }
            warming = false
        }
    }

    // Playback. 49 frames at 12 fps is about four seconds, which is the rate the clip is
    // meant to be seen at.
    LaunchedEffect(frames, playing, fps) {
        if (frames.isEmpty() || !playing) return@LaunchedEffect
        while (true) {
            delay(1000L / fps)
            frameIdx = (frameIdx + 1) % frames.size
        }
    }

    fun run(video: Boolean) {
        busy = true; image = null; frames = emptyList(); log = ""; progress = 0f
        status = if (video) "starting video" else "starting image"
        scope.launch {
            withContext(Dispatchers.IO) {
                try {
                    val s = seed.toLongOrNull() ?: 0L
                    if (video) {
                        val r = Video(ctx, runner) { say(it) }
                            .generate(prompt, s, srcImage) { what, i, n ->
                            status = what
                            if (n > 1) progress = i.toFloat() / n
                        }
                        frames = r.frames; frameIdx = 0; playing = true
                        status = "%d frames in %.1f s".format(r.frames.size, r.seconds)
                    } else {
                        val t0 = System.currentTimeMillis()
                        image = FirstFrame(ctx, runner) { say(it) }.generate(prompt, s) { i, n ->
                            status = "denoising $i/$n"; progress = i.toFloat() / n
                        }
                        status = "image in %.1f s".format(
                            (System.currentTimeMillis() - t0) / 1000.0)
                    }
                } catch (e: Throwable) {
                    status = "failed"
                    say("FAILED: " + e::class.simpleName + ": " + e.message)
                    showLog = true
                }
            }
            busy = false; progress = 0f; refresh++
        }
    }

    fun fetchModels() {
        settings.baseUrl = baseUrl
        settings.token = token
        showLog = true
        say("starting download service: ${gb(ModelStore.missingBytes(runner))}")
        // Deliberately NOT a coroutine in this composition. rememberCoroutineScope() is
        // tied to the composition, so backgrounding or recreating the Activity cancelled
        // the transfer mid-file -- which is what made it "error, then restart when I'm in
        // the app". A foreground service owns it now and survives both.
        DownloadService.start(ctx, baseUrl, token)
    }

    // Confirmation before an 8 GB download, showing what the device check actually
    // found. The compatibility result was previously only a banner and a disabled
    // button, which does not tell someone WHY -- and 8 GB is worth one explicit tap.
    if (confirmDownload) {
        val dc = devCheck
        val blocked = dc is DeviceCheck.Result.Unsupported || dc is DeviceCheck.Result.Fp16Suspect
        AlertDialog(
            onDismissRequest = { confirmDownload = false },
            title = { Text(if (blocked) "This device is not supported" else "Download models?") },
            text = {
                Column {
                    Text("Device: ${DeviceCheck.describeDevice()}", fontSize = 12.sp)
                    Spacer(Modifier.height(8.dp))
                    when (dc) {
                        is DeviceCheck.Result.Ok -> {
                            Text("✓  Hexagon HTP ${BuildConfig.HTP_ARCH} — context binaries load",
                                 fontSize = 12.sp,
                                 color = MaterialTheme.colorScheme.primary)
                            Text("✓  FP16 check passed (%.1f dB)".format(dc.snrDb),
                                 fontSize = 12.sp,
                                 color = MaterialTheme.colorScheme.primary)
                        }
                        is DeviceCheck.Result.Unsupported ->
                            Text("✗  The test graph would not load. Every model here " +
                                 "is compiled for Hexagon HTP ${BuildConfig.HTP_ARCH} (${BuildConfig.SOC_LABEL}), " +
                                 "so the download would be wasted.",
                                 fontSize = 12.sp, color = MaterialTheme.colorScheme.error)
                        is DeviceCheck.Result.Fp16Suspect ->
                            Text("✗  The test graph ran but returned wrong numbers " +
                                 "(%.1f dB). The float models would produce corrupt output."
                                 .format(dc.snrDb),
                                 fontSize = 12.sp, color = MaterialTheme.colorScheme.error)
                        else ->
                            Text("?  Compatibility could not be confirmed. You can try, " +
                                 "but the models may not load.", fontSize = 12.sp)
                    }
                    Spacer(Modifier.height(10.dp))
                    Text("${gb(ModelStore.missingBytes(runner))} over Wi-Fi. " +
                         "Resumes if interrupted.", fontSize = 12.sp)
                }
            },
            confirmButton = {
                Button(enabled = !blocked,
                       onClick = { confirmDownload = false; fetchModels() }) {
                    Text(if (blocked) "Unavailable" else "Download")
                }
            },
            dismissButton = {
                TextButton(onClick = { confirmDownload = false }) { Text("Cancel") }
            },
        )
    }

    // Scrollable: with the models panel open the content is taller than the screen, and
    // a fixed Column silently overlaps -- the output box sat on top of the generate
    // buttons rather than pushing them down.
    Column(
        Modifier.fillMaxSize()
            .statusBarsPadding()          // the title sat under the clock without this
            .navigationBarsPadding()      // ...and the log row under the gesture bar
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 10.dp)
    ) {

        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Neodragon Demo", style = MaterialTheme.typography.titleLarge,
                 softWrap = false, maxLines = 1)
            Spacer(Modifier.weight(1f))
            // softWrap=false: "rescan" was breaking across two lines once the status
            // text and both actions competed for the same row.
            TextButton(onClick = { showModels = !showModels }, enabled = !busy,
                       contentPadding = PaddingValues(horizontal = 10.dp)) {
                Text(if (showModels) "Close" else "Models", fontSize = 12.sp,
                     softWrap = false, maxLines = 1)
            }
            TextButton(onClick = { refresh++ }, enabled = !busy && !downloading,
                       contentPadding = PaddingValues(horizontal = 10.dp)) {
                Text("Rescan", fontSize = 12.sp, softWrap = false, maxLines = 1)
            }
        }

        // Status on its own line, where it has room.
        Text(
            if (missing.isEmpty()) "HTP ${BuildConfig.HTP_ARCH}  ·  ${ModelStore.MANIFEST.size} models ready"
            else "missing ${missing.size} of ${ModelStore.MANIFEST.size} models",
            style = MaterialTheme.typography.labelMedium,
            color = if (missing.isEmpty()) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.error
        )

        // ---- models ----------------------------------------------------------
        // Opens itself when models are absent, because with any missing the Image and
        // Video buttons are disabled and there is otherwise nothing to explain why.
        LaunchedEffect(missing.isEmpty()) { showModels = missing.isNotEmpty() }

        if (showModels) {
            Spacer(Modifier.height(8.dp))
            Column(
                Modifier.fillMaxWidth()
                    .clip(RoundedCornerShape(10.dp))
                    .background(MaterialTheme.colorScheme.surfaceContainerHighest)
                    .padding(12.dp)
            ) {
                Text(
                    if (missing.isEmpty()) "All ${ModelStore.MANIFEST.size} models present."
                    else "${missing.size} of ${ModelStore.MANIFEST.size} models missing " +
                         "(${gb(ModelStore.missingBytes(runner))} to fetch).",
                    fontSize = 12.sp
                )
                Spacer(Modifier.height(8.dp))
                OutlinedTextField(
                    value = baseUrl, onValueChange = { baseUrl = it },
                    label = { Text("base URL") }, singleLine = true,
                    enabled = !downloading,
                    textStyle = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(Modifier.height(4.dp))
                Text("Public repository — no sign-in needed.",
                     fontSize = 10.sp,
                     color = MaterialTheme.colorScheme.onSurfaceVariant)

                if (downloading) {
                    Spacer(Modifier.height(8.dp))
                    val overall =
                        if (dlTotal > 0) (dlBase + dlFileDone).toFloat() / dlTotal else 0f
                    Text(
                        "$dlName  ${gb(dlFileDone)} / ${gb(dlFileTotal)}" +
                        "   ~  overall ${(overall * 100).toInt()}%",
                        fontSize = 11.sp, fontFamily = FontFamily.Monospace
                    )
                    Spacer(Modifier.height(4.dp))
                    LinearProgressIndicator(
                        progress = { overall.coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth()
                    )
                }

                val dc = devCheck
                if (dc is DeviceCheck.Result.Unsupported || dc is DeviceCheck.Result.Fp16Suspect) {
                    Spacer(Modifier.height(10.dp))
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = MaterialTheme.colorScheme.errorContainer,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(Modifier.padding(10.dp)) {
                            Text(
                                if (dc is DeviceCheck.Result.Unsupported)
                                    "This device cannot run these models"
                                else "This device failed the FP16 check",
                                fontSize = 13.sp,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                            Spacer(Modifier.height(4.dp))
                            Text(
                                if (dc is DeviceCheck.Result.Unsupported)
                                    "Every graph here is compiled for Hexagon HTP ${BuildConfig.HTP_ARCH} " +
                                    "(${BuildConfig.SOC_LABEL}). A 58 KB test " +
                                    "graph would not load on ${DeviceCheck.describeDevice()}, " +
                                    "so the 8 GB download would be wasted."
                                else
                                    "The test graph ran but returned wrong numbers, so the " +
                                    "float models would produce corrupt output on " +
                                    "${DeviceCheck.describeDevice()}.",
                                fontSize = 11.sp,
                                color = MaterialTheme.colorScheme.onErrorContainer
                            )
                        }
                    }
                }

                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        enabled = !busy && !downloading && missing.isNotEmpty() &&
                                  baseUrl.isNotBlank() &&
                                  (devCheck?.canDownload ?: false),
                        onClick = {
                            if (Build.VERSION.SDK_INT >= 33) {
                                notifPerm.launch(android.Manifest.permission.POST_NOTIFICATIONS)
                            }
                            confirmDownload = true
                        },
                        modifier = Modifier.weight(1f)
                    ) { Text(if (missing.isEmpty()) "nothing to fetch" else "Download") }
                    if (downloading) {
                        OutlinedButton(onClick = { DownloadService.stop(ctx) }) { Text("Stop") }
                    }
                }
            }
        }

        Spacer(Modifier.height(8.dp))
        OutlinedTextField(
            value = prompt, onValueChange = { prompt = it },
            label = { Text("prompt") }, maxLines = 2,
            modifier = Modifier.fillMaxWidth()
        )

        // ---- image to video ---------------------------------------------------
        // With a photo chosen the SSD1B first frame is skipped entirely: ~2.1 s of
        // compute and 1.68 GB of context binaries that never need mapping.
        Spacer(Modifier.height(8.dp))
        Row(verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            val b = srcImage
            if (b == null) {
                OutlinedButton(
                    enabled = !busy && !downloading,
                    onClick = {
                        picker.launch(androidx.activity.result.PickVisualMediaRequest(
                            androidx.activity.result.contract.ActivityResultContracts
                                .PickVisualMedia.ImageOnly))
                    }
                ) { Text("Pick First Frame", fontSize = 12.sp) }
                Text("or leave empty to generate from the prompt",
                     fontSize = 11.sp,
                     color = MaterialTheme.colorScheme.onSurfaceVariant)
            } else {
                // Fixed 16:10 so the thumbnail is the shape the video will actually be,
                // rather than whatever the source aspect happened to be.
                // Fixed size, no aspect maths: the crop dialog always returns exactly
                // the model's input size, so 70x44 is already the right shape.
                Image(
                    bitmap = b.asImageBitmap(), contentDescription = "first frame",
                    modifier = Modifier.size(width = 70.dp, height = 44.dp)
                        .clip(RoundedCornerShape(6.dp)),
                    contentScale = ContentScale.Crop
                )
                Column(Modifier.weight(1f)) {
                    Text("This photo is frame 1", fontSize = 11.sp)
                    Text("${b.width}×${b.height}", fontSize = 10.sp,
                         color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                TextButton(enabled = !busy, onClick = {
                    picker.launch(androidx.activity.result.PickVisualMediaRequest(
                        androidx.activity.result.contract.ActivityResultContracts
                            .PickVisualMedia.ImageOnly))
                }) { Text("Change", fontSize = 11.sp) }
                TextButton(enabled = !busy, onClick = { srcImage = null }) {
                    Text("Remove", fontSize = 11.sp)
                }
            }
        }

        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = seed, onValueChange = { seed = it.filter(Char::isDigit).take(9) },
                label = { Text("seed") }, singleLine = true,
                modifier = Modifier.width(104.dp)
            )
        }

        // The two generate actions get their own full-width row: "Generate first frame
        // only" wrapped to three lines when it shared a row with the seed field.
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.fillMaxWidth()) {
            Button(enabled = !busy && !downloading && missing.isEmpty() && srcImage == null,
                   onClick = { run(false) },
                   modifier = Modifier.weight(1f)) {
                Text("First Frame Only", fontSize = 12.sp, softWrap = false, maxLines = 1)
            }
            Button(enabled = !busy && !downloading && missing.isEmpty(),
                   onClick = { run(true) },
                   modifier = Modifier.weight(1f)) {
                Text("Generate Video", fontSize = 12.sp, softWrap = false, maxLines = 1)
            }
        }

        if (warming && !busy) {
            Spacer(Modifier.height(6.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp)
                Spacer(Modifier.width(8.dp))
                Text("Warming up models (~5 s) — generating now will wait for this",
                     fontSize = 11.sp,
                     color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }

        if (busy || status.isNotEmpty()) {
            Spacer(Modifier.height(8.dp))
            Text(status, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            if (busy) {
                Spacer(Modifier.height(4.dp))
                if (progress > 0f) {
                    LinearProgressIndicator(progress = { progress },
                                            modifier = Modifier.fillMaxWidth())
                } else {
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                }
            }
        }

        Spacer(Modifier.height(10.dp))

        // ---- output ----------------------------------------------------------
        // The box keeps the clip's 512x320 aspect at full width so the layout does not
        // jump when the first frame arrives, and so the placeholder occupies the space
        // the video will.
        val shown = if (frames.isNotEmpty())
            frames[frameIdx.coerceIn(0, frames.size - 1)] else image
        val ar = if (shown != null) shown.width.toFloat() / shown.height.toFloat()
                 else 512f / 320f
        Box(
            Modifier.fillMaxWidth()
                .aspectRatio(ar)
                .clip(RoundedCornerShape(10.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant),
            contentAlignment = Alignment.Center
        ) {
            if (shown != null) {
                Image(
                    bitmap = shown.asImageBitmap(), contentDescription = "output",
                    modifier = Modifier.fillMaxSize(), contentScale = ContentScale.Fit
                )
            } else {
                Text("no output yet", fontSize = 12.sp)
            }
        }

        if (frames.isNotEmpty()) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                TextButton(onClick = { playing = !playing },
                           modifier = Modifier.width(64.dp)) {
                    Text(if (playing) "Pause" else "Play", fontSize = 11.sp)
                }
                Text("${frameIdx + 1}/${frames.size}", fontSize = 11.sp,
                     modifier = Modifier.width(50.dp))
                Slider(
                    value = frameIdx.toFloat(),
                    onValueChange = { playing = false; frameIdx = it.toInt() },
                    valueRange = 0f..(frames.size - 1).toFloat(),
                    modifier = Modifier.weight(1f)
                )
                TextButton(onClick = {
                    fps = when (fps) { 12 -> 24; 24 -> 6; else -> 12 }
                }, modifier = Modifier.width(64.dp)) { Text("${fps}fps", fontSize = 11.sp) }
                TextButton(enabled = !busy && !saving, onClick = {
                    saving = true; savedMsg = ""
                    scope.launch {
                        withContext(Dispatchers.IO) {
                            try {
                                // Encode at 24 fps regardless of the preview rate: the
                                // clip is 49 frames of a 2 s shot, and the fps toggle is
                                // a viewing aid, not a property of the video.
                                VideoWriter.write(ctx, frames, 24, log = { say(it) })
                                savedMsg = "Saved to Movies/Neodragon"
                            } catch (e: Throwable) {
                                savedMsg = "Save failed"
                                say("SAVE FAILED: " + e::class.simpleName + ": " + e.message)
                                showLog = true
                            }
                        }
                        saving = false
                    }
                }) { Text(if (saving) "Saving…" else "Save MP4", fontSize = 11.sp) }
            }
        }

        if (savedMsg.isNotEmpty()) {
            Text(savedMsg, fontSize = 11.sp, color = MaterialTheme.colorScheme.primary)
        }

        Spacer(Modifier.height(16.dp))

        // ---- log -------------------------------------------------------------
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(onClick = { showLog = !showLog }) {
                Text(if (showLog) "Hide Log" else "Show Log", fontSize = 12.sp)
            }
            Spacer(Modifier.weight(1f))
            TextButton(enabled = !busy, onClick = {
                showLog = true; busy = true; log = ""
                scope.launch {
                    withContext(Dispatchers.IO) {
                        try {
                            for (n in listOf("cliplp", "distilt5f", "ctxadaptfp16",
                                             "vaeenc", "vaedecsn", "mmdit_s0g")) {
                                if (!runner.isReady(n)) { say(n + ": absent"); continue }
                                say(n)
                                runner.describe(n).split(";").filter { it.isNotBlank() }
                                    .forEach { say("   " + it) }
                                runner.release(n)
                            }
                        } catch (e: Throwable) { say("probe failed: " + e.message) }
                    }
                    busy = false
                }
            }) { Text("Probe Graphs", fontSize = 12.sp) }
        }

        if (showLog) {
            val scroll = rememberScrollState()
            LaunchedEffect(log) { scroll.animateScrollTo(scroll.maxValue) }
            // Fixed height, and the weight(1f) spacer above is what yields space to it.
            // Previously the panel shared a flexible column with the output box and the
            // status rows, so opening the models panel or starting a save visibly
            // shrank the log while you were reading it.
            Box(
                Modifier.fillMaxWidth().height(300.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(MaterialTheme.colorScheme.surfaceContainerHighest)
                    .verticalScroll(scroll)
                    .padding(8.dp)
            ) {
                // SelectionContainer makes the log long-press selectable and copyable --
                // these timings are the actual measurements, and retyping them off a
                // screen is not a reasonable way to get them off the phone.
                androidx.compose.foundation.text.selection.SelectionContainer {
                    Text(log.ifBlank { "ready." }, fontFamily = FontFamily.Monospace,
                         fontSize = 10.sp, lineHeight = 13.sp)
                }
            }
        }
    }
}

/** Bytes as GB/MB, for progress lines where three significant figures is plenty. */
private fun gb(b: Long): String = when {
    b >= 1L shl 30 -> "%.2f GB".format(b.toDouble() / (1L shl 30))
    b >= 1L shl 20 -> "%.0f MB".format(b.toDouble() / (1L shl 20))
    else -> "%d B".format(b)
}
