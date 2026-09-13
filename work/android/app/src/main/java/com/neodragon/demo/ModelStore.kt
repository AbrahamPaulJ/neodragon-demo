package com.neodragon.demo

import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Downloads the converted context binaries into the app's external files dir.
 *
 * The models are ~7.9 GB in total and cannot ship inside the APK, so they are fetched on
 * first run from a base URL the user supplies in the UI. Each file is verified by BYTE
 * LENGTH against the manifest and re-downloaded if short -- large transfers to this
 * device silently truncated twice during the port (a 1.5 GB push arrived as 113 MB and a
 * 1.4 GB one as 577 MB, both reported success), and the failure surfaces much later as an
 * unhelpful "Create From Binary failure".
 *
 * Downloads resume with an HTTP Range request so a dropped connection does not restart a
 * 1.5 GB file.
 *
 * ## Why redirects are followed by hand
 *
 * The models live in a PRIVATE Hugging Face repo, so the request to
 * `huggingface.co/.../resolve/main/x.bin` needs an `Authorization: Bearer <token>`
 * header. That endpoint answers **302** with a *pre-signed* URL on a different host
 * (`us.aws.cdn.hf.co`), where the credential is in the query string.
 *
 * The token must not survive that hop. Measured against the live repo
 * (`work/device/check_hf_download.py`): the CDN in fact accepts the redundant header and
 * still returns 206, so this is **not** about the request failing -- it is that following
 * a redirect blindly hands a bearer token to whatever host the `Location` names. Java's
 * automatic redirect handling gives no way to drop a header mid-chain, so
 * `instanceFollowRedirects` is turned off and the chain is walked here:
 *
 *  * `Authorization` is attached only when the host is [AUTH_HOST] -- so it reaches
 *    neither the CDN nor, if someone edits the base URL in the UI, an arbitrary server.
 *  * `Range` IS carried across every hop, or a resumed download would restart from zero
 *    and append a second copy onto the partial file.
 *
 * Verified end to end against the live repo: unauthenticated 401, authenticated 302 to a
 * signed CDN URL, CDN + `Range` without auth 206, and the returned bytes equal to the
 * locally built artefact.
 */
object ModelStore {

    /** The only host the bearer token is ever sent to. */
    private const val AUTH_HOST = "huggingface.co"

    /** Prefilled in the UI. Set at build time with -PmodelBaseUrl (build.gradle.kts). */
    val DEFAULT_BASE_URL: String = BuildConfig.MODEL_BASE_URL

    /**
     * name -> expected byte size. Sizes are from the v79 artefacts: a build for another
     * HTP revision produces different sizes, so update this list with the new binaries.
     */
    val MANIFEST: List<Pair<String, Long>> = listOf(
        // video path -- the fused-score rebuilds (session 7, trap #40). They replace
        // mmdit_s0g/s1f/s2f: +10.9/+12.2/+12.0 dB, every stage past its paper target for
        // the first time, and 4.4% fewer accelerator cycles. Anyone holding the old set
        // re-downloads, because the graphs genuinely changed.
        "mmdit_s0fs" to 1_525_163_128L,
        "mmdit_s1fs" to 1_533_523_064L,
        "mmdit_s2fs" to 1_573_344_888L,
        // 2x upscale, 320x512 -> 640x1024. Small enough that it is not worth making
        // optional in the manifest, and without it the video is not the resolution the
        // paper's headline describes.
        "quicksrm2x" to 377_880L,
        "ctxadaptfp16" to 260_531_784L,
        "distilt5f" to 260_050_504L,
        "vaeenc" to 41_667_768L,
        "vaedecsn" to 12_104_632L,
        // first-frame path
        "clipl" to 233_993_824L,
        "cliplp" to 249_451_112L,
        "clipg" to 1_402_277_416L,
        "ssd1bunet" to 1_358_184_480L,
        "ssd1bvaedec" to 85_467_216L,
    )

    fun totalBytes() = MANIFEST.sumOf { it.second }

    fun missing(runner: QnnRunner): List<Pair<String, Long>> =
        MANIFEST.filter { (n, sz) -> runner.contextFile(n).length() != sz }

    fun missingBytes(runner: QnnRunner): Long = missing(runner).sumOf { it.second }

    /**
     * Open [url0], following redirects manually. [token] is sent only to [AUTH_HOST];
     * [rangeFrom] > 0 adds a Range header that is re-sent on every hop.
     *
     * Returns a connected [HttpURLConnection] whose response is 200 or 206.
     */
    private fun open(url0: String, token: String, rangeFrom: Long): HttpURLConnection {
        var url = URL(url0)
        var hops = 0
        while (true) {
            val current = url
            val c = (current.openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = 30_000
                readTimeout = 60_000
                if (token.isNotBlank() && current.host.equals(AUTH_HOST, ignoreCase = true)) {
                    setRequestProperty("Authorization", "Bearer " + token)
                }
                if (rangeFrom > 0) setRequestProperty("Range", "bytes=" + rangeFrom + "-")
            }
            val code = c.responseCode
            if (code == HttpURLConnection.HTTP_MOVED_PERM ||
                code == HttpURLConnection.HTTP_MOVED_TEMP ||
                code == HttpURLConnection.HTTP_SEE_OTHER ||
                code == 307 || code == 308
            ) {
                val loc = c.getHeaderField("Location")
                c.disconnect()
                check(loc != null) { "redirect " + code + " with no Location" }
                check(++hops <= 5) { "too many redirects" }
                url = URL(current, loc)      // resolves relative Locations too
                continue
            }
            if (code == HttpURLConnection.HTTP_OK || code == HttpURLConnection.HTTP_PARTIAL) {
                return c
            }
            // Read the body: HF puts the real reason there ("Invalid credentials", etc).
            val why = runCatching { c.errorStream?.bufferedReader()?.readText() }
                .getOrNull().orEmpty().take(300)
            c.disconnect()
            val what = when (code) {
                401 -> "401 unauthorized - token missing, wrong, or lacks read access"
                403 -> "403 forbidden - the token cannot read this repo"
                404 -> "404 not found - check the base URL and the file name"
                416 -> "416 range not satisfiable - local file longer than the remote"
                else -> "HTTP " + code
            }
            error(what + if (why.isBlank()) "" else " :: " + why)
        }
    }

    /**
     * Fetch every missing model. [onProgress] receives (name, bytesDone, bytesTotal).
     */
    fun download(
        runner: QnnRunner,
        baseUrl: String,
        token: String = "",
        onProgress: (String, Long, Long) -> Unit,
        shouldStop: () -> Boolean = { false },
    ) {
        for ((name, size) in missing(runner)) {
            val dst = runner.contextFile(name)

            // A file longer than the manifest says is not a resumable partial -- it is a
            // different artefact. Start over rather than Range-requesting past the end.
            if (dst.exists() && dst.length() > size) dst.delete()
            var have = if (dst.exists()) dst.length() else 0L

            var attempt = 0
            while (dst.length() != size) {
                if (shouldStop()) return
                attempt++
                check(attempt <= 5) { name + ": gave up after " + attempt + " attempts" }
                onProgress(name, have, size)

                val c = open(baseUrl.trimEnd('/') + "/" + name + "_" + BuildConfig.HTP_ARCH + ".bin", token, have)
                try {
                    // 200 means the server ignored the Range and is sending the whole
                    // file, so the partial on disk must be overwritten, not appended to.
                    val append = c.responseCode == HttpURLConnection.HTTP_PARTIAL
                    if (!append) have = 0
                    c.inputStream.use { ins ->
                        FileOutputStream(dst, append).use { fo ->
                            val buf = ByteArray(1 shl 20)
                            while (true) {
                                if (shouldStop()) return
                                val n = ins.read(buf)
                                if (n <= 0) break
                                fo.write(buf, 0, n)
                                have += n
                                onProgress(name, have, size)
                            }
                        }
                    }
                } catch (e: java.io.IOException) {
                    // A dropped connection is expected on a multi-GB transfer; fall
                    // through and resume from whatever actually landed on disk.
                    if (attempt >= 5) throw e
                } finally {
                    c.disconnect()
                }
                have = dst.length()          // short read -> resume from the truth on disk
            }
            onProgress(name, size, size)
        }
    }
}
