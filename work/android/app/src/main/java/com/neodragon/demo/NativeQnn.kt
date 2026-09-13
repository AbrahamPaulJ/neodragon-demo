package com.neodragon.demo

/**
 * Thin JNI surface onto libndqnn.so. See ndqnn.cpp for why the app runs QNN in-process
 * instead of exec'ing the bundled qnn-net-run.
 *
 * Every call is by tensor NAME, matched against the context binary's own metadata, so a
 * graph whose inputs the converter reordered cannot silently be fed the wrong tensor.
 */
object NativeQnn {

    @Volatile private var loaded = false

    fun ensureLoaded() {
        if (!loaded) { System.loadLibrary("ndqnn"); loaded = true }
    }

    @JvmStatic external fun init(backendLib: String, systemLib: String, skelDir: String): Boolean
    @JvmStatic external fun load(binPath: String): Long
    @JvmStatic external fun describe(handle: Long): String
    @JvmStatic external fun outputNames(handle: Long): Array<String>
    @JvmStatic external fun execute(
        handle: Long, names: Array<String>, data: Array<FloatArray>,
    ): Array<FloatArray>?
    @JvmStatic external fun free(handle: Long)
    @JvmStatic external fun lastError(): String
}
