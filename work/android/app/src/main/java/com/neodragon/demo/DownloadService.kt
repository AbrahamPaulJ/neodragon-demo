package com.neodragon.demo

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import kotlin.concurrent.thread

/**
 * The model download, as a foreground service.
 *
 * It used to run in the Activity's `rememberCoroutineScope()`. That scope is tied to the
 * COMPOSITION: the moment Android backgrounded or recreated the Activity the scope was
 * cancelled and the transfer died -- which is why it "errors and can be restarted from
 * inside the app". Android also throttles network for processes that are not foreground,
 * so even surviving cancellation would not have been enough.
 *
 * A foreground service with an ongoing notification is the sanctioned way to hold a
 * multi-minute transfer. ~8 GB over Wi-Fi is exactly the case it exists for.
 *
 * Progress is published through [State], a process-wide singleton the Compose UI reads,
 * rather than through binding: the UI may not exist while the service runs, and must be
 * able to attach to an already-running download when it comes back.
 */
class DownloadService : Service() {

    /** Process-wide download state. Survives the Activity; the UI observes it. */
    object State {
        @Volatile var running = false
        @Volatile var name = ""
        @Volatile var doneBytes = 0L
        @Volatile var totalBytes = 0L
        /** Size of the file currently in flight. */
        @Volatile var totalFileBytes = 0L
        @Volatile var message = ""
        @Volatile var finished = false
        /** Set by the UI to ask the service to stop at the next chunk boundary. */
        @Volatile var stopRequested = false
    }

    companion object {
        private const val CHANNEL = "nd_download"
        private const val NOTIF_ID = 4711
        const val ACTION_START = "com.neodragon.demo.DOWNLOAD_START"
        const val ACTION_STOP = "com.neodragon.demo.DOWNLOAD_STOP"
        const val EXTRA_BASE_URL = "baseUrl"
        const val EXTRA_TOKEN = "token"

        fun start(ctx: Context, baseUrl: String, token: String) {
            val i = Intent(ctx, DownloadService::class.java).apply {
                action = ACTION_START
                putExtra(EXTRA_BASE_URL, baseUrl)
                putExtra(EXTRA_TOKEN, token)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) ctx.startForegroundService(i)
            else ctx.startService(i)
        }

        fun stop(ctx: Context) {
            State.stopRequested = true
            ctx.startService(Intent(ctx, DownloadService::class.java).apply {
                action = ACTION_STOP
            })
        }
    }

    private lateinit var runner: QnnRunner
    private var worker: Thread? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        runner = QnnRunner(this)
        ensureChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            State.stopRequested = true
            return START_NOT_STICKY
        }
        if (State.running) return START_NOT_STICKY      // already going; ignore re-entry

        val baseUrl = intent?.getStringExtra(EXTRA_BASE_URL).orEmpty()
        val token = intent?.getStringExtra(EXTRA_TOKEN).orEmpty()

        startForeground(NOTIF_ID, build("Starting…", "", 0))

        State.running = true
        State.finished = false
        State.stopRequested = false
        State.message = "starting"
        State.doneBytes = 0
        State.totalBytes = ModelStore.missingBytes(runner)

        worker = thread(name = "nd-download") {
            var base = 0L
            var last = ""
            var lastPaint = 0L
            try {
                ModelStore.download(
                    runner, baseUrl, token,
                    onProgress = { n, done, total ->
                        if (n != last) {
                            if (last.isNotEmpty()) base += State.totalFileBytes
                            last = n
                            State.name = n
                        }
                        State.totalFileBytes = total
                        State.doneBytes = base + done
                        // Repaint the notification at 4 MB granularity: the callback
                        // fires per 1 MB chunk, ~8000 times over the full download.
                        if (State.doneBytes - lastPaint >= (4L shl 20)) {
                            lastPaint = State.doneBytes
                            notify(build(n, "${State.doneBytes shr 20} / " +
                                            "${State.totalBytes shr 20} MB",
                                         pct(State.doneBytes, State.totalBytes)))
                        }
                    },
                    shouldStop = { State.stopRequested },
                )
                State.message = if (State.stopRequested)
                    "Stopped — resumes where it left off" else "All models downloaded"
            } catch (e: Throwable) {
                State.message = "Failed: ${e.message}"
            } finally {
                State.running = false
                State.finished = true
                notifyFinal(State.message)
                stopForeground(STOP_FOREGROUND_DETACH)
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        State.stopRequested = true
        super.onDestroy()
    }

    // ---- notification ------------------------------------------------------

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val ch = NotificationChannel(CHANNEL, "Model download",
                                     NotificationManager.IMPORTANCE_LOW).apply {
            description = "Progress while the Neodragon models download"
            setShowBadge(false)
        }
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
            .createNotificationChannel(ch)
    }

    private fun contentIntent(): PendingIntent =
        PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)

    private fun build(title: String, text: String, pct: Int): Notification =
        Notification.Builder(this, CHANNEL)
            .setContentTitle("Downloading models  $pct%")
            .setContentText(if (text.isEmpty()) title else "$title  $text")
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setProgress(100, pct, false)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(contentIntent())
            .build()

    private fun notify(n: Notification) {
        runCatching {
            (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).notify(NOTIF_ID, n)
        }
    }

    private fun notifyFinal(msg: String) {
        runCatching {
            val n = Notification.Builder(this, CHANNEL)
                .setContentTitle("Neodragon")
                .setContentText(msg)
                .setSmallIcon(android.R.drawable.stat_sys_download_done)
                .setOngoing(false)
                .setAutoCancel(true)
                .setContentIntent(contentIntent())
                .build()
            (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).notify(NOTIF_ID, n)
        }
    }

    private fun pct(done: Long, total: Long) =
        if (total > 0) ((done * 100) / total).toInt().coerceIn(0, 100) else 0
}
