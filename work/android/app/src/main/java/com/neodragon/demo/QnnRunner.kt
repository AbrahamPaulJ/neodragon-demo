package com.neodragon.demo

import android.content.Context
import java.io.File

/**
 * Runs converted QNN context binaries on the HTP, in-process.
 *
 * The first version of this class exec'd the bundled `qnn-net-run` exactly as the desktop
 * harness does over adb. That cannot work from inside an app: the byte-identical binary
 * (verified by md5, with the same libraries, skel directory and working directory) loads
 * the DSP skel from /data/local/tmp and fails from /data/app with `loadRemoteSymbols
 * failed with err 4000`, because the executable's SELinux context is `apk_data_file`
 * rather than `shell_data_file` and the resulting process is denied the Hexagon fastrpc
 * device. The backend is therefore dlopen'd into this process instead -- see ndqnn.cpp.
 *
 * Two consequences worth knowing:
 *
 *  * **Graphs stay resident.** [load] is paid once per model, not once per inference.
 *    That matters most for the video path, where the AR loop visits all three ~1.5 GB
 *    MMDiT stages every unit; the residency probe showed all three co-resident in
 *    3.58 GB with ~1.5 GB spare.
 *  * **No more input-format footgun.** The exec path needed `--use_native_input_files`
 *    for integer inputs to float graphs and had to omit it for quantised graphs, and
 *    getting it wrong produced identical plausible output for every prompt (trap #8).
 *    In-process, the native side reads each tensor's real dtype and encoding out of the
 *    binary and converts accordingly, so there is nothing left to get wrong.
 */
class QnnRunner(private val ctx: Context) {

    private val nativeDir: String = ctx.applicationInfo.nativeLibraryDir

    val root: File = File(ctx.getExternalFilesDir(null), "nd").apply { mkdirs() }
    val ctxDir: File = File(root, "ctx").apply { mkdirs() }

    /** Graph metadata is read from the binary itself; only the file name is needed here. */
    data class Graph(
        val name: String,
        val inputs: List<String> = emptyList(),
        val outputs: List<String> = emptyList(),
        val nativeInput: Boolean = false,   // vestigial: the exec path needed it, JNI does not
    )

    private val handles = HashMap<String, Long>()

    @Volatile private var initialised = false

    fun contextFile(name: String) = File(ctxDir, "${name}_${BuildConfig.HTP_ARCH}.bin")

    fun isReady(name: String) = contextFile(name).let { it.exists() && it.length() > 0 }

    /** Bring up the backend and the DSP. Safe to call repeatedly. */
    @Synchronized
    fun initBackend() {
        if (initialised) return
        NativeQnn.ensureLoaded()
        val ok = NativeQnn.init(
            File(nativeDir, "libQnnHtp.so").absolutePath,
            File(nativeDir, "libQnnSystem.so").absolutePath,
            nativeDir,           // holds libQnnHtpV79Skel.so, found via ADSP_LIBRARY_PATH
        )
        check(ok) { "QNN init failed: ${NativeQnn.lastError()}" }
        initialised = true
    }

    /** Load (and cache) a model, returning its native handle. */
    @Synchronized
    fun handleOf(name: String): Long {
        initBackend()
        handles[name]?.let { return it }
        val f = contextFile(name)
        check(f.exists()) { "model $name is not on the device (${f.absolutePath})" }
        val h = NativeQnn.load(f.absolutePath)
        check(h != 0L) { "load $name failed: ${NativeQnn.lastError()}" }
        handles[name] = h
        return h
    }

    /** Human-readable tensor list straight out of the binary; useful when wiring a graph. */
    fun describe(name: String): String = NativeQnn.describe(handleOf(name))

    /**
     * Execute [g] with named real-valued tensors. Integer inputs (token ids) are passed as
     * floats -- ids up to 49407 are exact in fp32 -- and the native side writes them in the
     * graph's actual dtype.
     */
    fun run(
        g: Graph,
        floats: Map<String, FloatArray> = emptyMap(),
        intInputs: Map<String, IntArray> = emptyMap(),
    ): Map<String, FloatArray> {
        val h = handleOf(g.name)

        val all = LinkedHashMap<String, FloatArray>(floats)
        intInputs.forEach { (k, v) -> all[k] = FloatArray(v.size) { v[it].toFloat() } }

        val names = all.keys.toTypedArray()
        val data = names.map { all.getValue(it) }.toTypedArray()

        val out = NativeQnn.execute(h, names, data)
            ?: error("${g.name}: execute failed: ${NativeQnn.lastError()}")
        val outNames = NativeQnn.outputNames(h)
        return outNames.indices.associate { outNames[it] to out[it] }
    }

    @Synchronized
    fun release(name: String) {
        handles.remove(name)?.let { NativeQnn.free(it) }
    }

    @Synchronized
    fun releaseAll() {
        handles.values.forEach { NativeQnn.free(it) }
        handles.clear()
    }
}
