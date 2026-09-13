package com.neodragon.demo

import android.content.Context
import android.os.Build
import java.io.File
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.sqrt

/**
 * Decide whether this phone can run the pipeline BEFORE offering an 8 GB download.
 *
 * Every context binary here is compiled for one Hexagon revision ([BuildConfig.HTP_ARCH],
 * v79 / SM8750 by default). On anything older the binary will not deserialize, and the user finds
 * that out after downloading 8 GB, as an unhelpful "Create From Binary failure".
 *
 * The check is a **canary graph shipped inside the APK** (58 KB), not a `Build.SOC_MODEL`
 * allowlist. A string match is a guess about which SoCs carry which Hexagon revision, and
 * it goes stale with every new chip; loading and running a real context binary proves the
 * backend, the skel and the architecture actually agree.
 *
 * It also checks **fp16 arithmetic**, because two shipping graphs (`ctxadaptfp16`,
 * `distilt5f`) are float and the HTP runs float graphs in fp16 with no fp32 upcast in the
 * layer norm (trap #3 -- an unscaled DistilT5 build once returned 74% NaN on device). A
 * device where fp16 misbehaves produces plausible-looking garbage rather than an error,
 * so the canary compares against a reference computed on the host at fp32.
 */
object DeviceCheck {

    /** Below this SNR against the host reference, fp16 is not doing what it should. */
    private const val MIN_SNR_DB = 20.0

    sealed class Result {
        /** Canary loaded and its output matches. */
        data class Ok(val snrDb: Double) : Result()

        /** The context binary would not load: wrong Hexagon revision, almost always. */
        data class Unsupported(val detail: String) : Result()

        /** Loaded and ran, but the numbers are wrong -- fp16 path is suspect. */
        data class Fp16Suspect(val snrDb: Double) : Result()

        /** Could not reach a verdict (asset missing, backend refused to init). */
        data class Inconclusive(val detail: String) : Result()

        val canDownload: Boolean get() = this is Ok || this is Inconclusive
    }

    fun describeDevice(): String =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S)
            "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}"
        else "${Build.MANUFACTURER} ${Build.MODEL}"

    /**
     * Run the canary. Cheap (a few ms of compute) but it does bring the backend up, so
     * call it off the main thread.
     */
    fun run(ctx: Context, runner: QnnRunner): Result {
        val canary = "canary_${BuildConfig.HTP_ARCH}.bin"
        val bin = File(ctx.cacheDir, canary)
        try {
            if (!bin.exists() || bin.length() == 0L) {
                ctx.assets.open(canary).use { ins ->
                    bin.outputStream().use { ins.copyTo(it) }
                }
            }
            val x = readFloats(ctx, "canary_in.raw")
            val ref = readFloats(ctx, "canary_ref.raw")

            runner.initBackend()

            val h = NativeQnn.load(bin.absolutePath)
            if (h == 0L) {
                // This is the "wrong chip" path. The message from QNN is not
                // user-facing prose, so it is kept for the log and summarised above it.
                return Result.Unsupported(NativeQnn.lastError())
            }
            try {
                val out = NativeQnn.execute(h, arrayOf("x"), arrayOf(x))
                    ?: return Result.Inconclusive("execute failed: ${NativeQnn.lastError()}")
                val got = out.firstOrNull { it.size == ref.size }
                    ?: return Result.Inconclusive("canary returned no tensor of ${ref.size}")

                // NaN/Inf is the trap #3 signature and must not be allowed to become a
                // large-but-finite SNR by accident.
                if (got.any { it.isNaN() || it.isInfinite() }) return Result.Fp16Suspect(-1.0)

                val snr = snrDb(ref, got)
                return if (snr >= MIN_SNR_DB) Result.Ok(snr) else Result.Fp16Suspect(snr)
            } finally {
                NativeQnn.free(h)
            }
        } catch (e: Throwable) {
            // Deliberately permissive: a check that cannot run must not block a device
            // that would otherwise work. Unsupported is only returned on a real refusal.
            return Result.Inconclusive("${e::class.simpleName}: ${e.message}")
        }
    }

    private fun readFloats(ctx: Context, name: String): FloatArray {
        val b = ctx.assets.open(name).use { it.readBytes() }
        val bb = java.nio.ByteBuffer.wrap(b).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        return FloatArray(b.size / 4) { bb.getFloat(it * 4) }
    }

    private fun snrDb(ref: FloatArray, got: FloatArray): Double {
        var num = 0.0
        var den = 0.0
        for (i in ref.indices) {
            num += ref[i].toDouble() * ref[i]
            val d = ref[i].toDouble() - got[i]
            den += d * d
        }
        if (den == 0.0) return Double.POSITIVE_INFINITY
        return 20.0 * log10(sqrt(num) / sqrt(den))
    }
}
