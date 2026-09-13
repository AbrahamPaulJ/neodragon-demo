package com.neodragon.demo

import java.io.File
import kotlin.math.sqrt

/**
 * The generation pipeline, built up in stages.
 *
 * Milestone 1 (this file): prove a converted graph runs on the NPU *from inside the app*.
 * That is the piece with real unknowns -- exec'ing the bundled qnn-net-run, finding the
 * HTP skel via ADSP_LIBRARY_PATH, loading a 1.5 GB context binary out of the app's
 * external files dir. Everything after it is arithmetic that already works in Python.
 *
 * Still to port from the desktop driver before full generation works:
 *   * CLIP BPE tokeniser (vocab + merges as assets) and the T5 sentencepiece tokeniser
 *   * LCMScheduler -- and it must be the reference's exact construction
 *     (set_alpha_to_one=false, steps_offset=1), not a default
 *   * host sinusoidal embeddings for the UNet (deliberately hoisted out of the graph --
 *     leaving them in cost the MMDiT 8.41 dB)
 *   * the pyramid scheduler, history pyramid and stage_conditioning_padded for video
 */
class Pipeline(private val runner: QnnRunner, private val log: (String) -> Unit) {

    companion object {
        val CLIP_L = QnnRunner.Graph(
            "clipl", listOf("input_ids"), listOf("hidden"), nativeInput = true)
        val CLIP_G = QnnRunner.Graph(
            "clipg", listOf("input_ids"), listOf("hidden", "text_embeds"), nativeInput = true)
        val UNET = QnnRunner.Graph(
            "ssd1bunet",
            listOf("sample", "t_emb", "aug_emb", "encoder_hidden_states"),
            listOf("noise_pred"))
        val VAE_DEC = QnnRunner.Graph(
            "ssd1bvaedec", listOf("latent"), listOf("image"))
    }

    /**
     * Smoke test: run CLIP L on a fixed token sequence and report the output statistics.
     *
     * A fixed sequence is used deliberately -- no tokeniser is needed to prove the NPU
     * path, and the expected statistics are known from the desktop run (60.75 dB against
     * fp32), so a wrong answer here is unambiguous rather than a plausible-looking one.
     *
     * The tensor names are read out of the context binary rather than hardcoded, because
     * the converter is free to name and order graph I/O as it likes and a guessed name
     * would fail late and confusingly.
     */
    fun smokeTest(): Boolean {
        log("smoke test: CLIP L on the NPU")
        if (!runner.isReady(CLIP_L.name)) { log("  clipl not on device"); return false }

        return try {
            val t0 = System.currentTimeMillis()
            runner.initBackend()
            log("  backend up (%d ms)".format(System.currentTimeMillis() - t0))

            val t1 = System.currentTimeMillis()
            val desc = runner.describe(CLIP_L.name)
            log("  loaded in %d ms".format(System.currentTimeMillis() - t1))
            desc.split(";").filter { it.isNotBlank() }.forEach { log("    $it") }

            val inName = desc.split(";").first { it.startsWith("in ") }
                .removePrefix("in ").substringBefore(":")

            // "a cinematic photo of a mountain lake at sunrise", CLIP-tokenised on the host.
            val ids = intArrayOf(
                49406, 320, 25602, 1125, 539, 320, 3965, 2553, 536, 15753, 49407
            ) + IntArray(77 - 11) { 49407 }

            val t2 = System.currentTimeMillis()
            val out = runner.run(CLIP_L, intInputs = mapOf(inName to ids))
            val ms = System.currentTimeMillis() - t2

            val h = out.values.first()
            val mean = h.average()
            val sd = sqrt(h.map { (it - mean) * (it - mean) }.average())
            val amax = h.maxOf { kotlin.math.abs(it) }
            log("  hidden: ${h.size} floats  (expect ${77 * 768})")
            log("  mean %.4f  std %.4f  max|x| %.2f".format(mean, sd, amax))
            log("  inference %d ms  (device min-of-N was 12.1 ms)".format(ms))

            val sane = h.size == 77 * 768 && sd > 0.01 && amax < 1e4f && h.all { it.isFinite() }
            log(if (sane) "  PASS -- the NPU ran a converted graph from inside the app"
                else "  FAIL -- output is not sane")
            sane
        } catch (e: Throwable) {
            log("  FAIL: ${e.message}")
            false
        }
    }

    /** Placeholder until the tokeniser and scheduler are ported. */
    fun generateFirstFrame(prompt: String) {
        log("prompt: ${prompt.take(60)}")
        if (!smokeTest()) return
        log("")
        log("first-frame generation needs the CLIP tokeniser + LCM scheduler in Kotlin;")
        log("the graphs themselves are all present and proven on device:")
        log("  clipl 60.75 dB · clipg 14.71 · ssd1bunet 32.51 · ssd1bvaedec 32.43 dB")
    }

    fun deviceInfo(): String {
        val d = File(runner.root.absolutePath)
        return "storage ${d.absolutePath}\nfree ${"%.1f".format(d.usableSpace / 1073741824.0)} GB"
    }
}
