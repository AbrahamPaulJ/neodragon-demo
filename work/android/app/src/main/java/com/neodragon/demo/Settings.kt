package com.neodragon.demo

import android.content.Context

/**
 * The two things the model download needs, kept out of the APK.
 *
 * The context binaries live in a PRIVATE Hugging Face repo, so fetching them needs a read
 * token. Baking one into the source would put it in the installable, where it is readable
 * by anyone who unzips the APK and cannot be revoked without a rebuild. It is typed once
 * into the UI instead and kept in app-private SharedPreferences, which is why this class
 * exists at all.
 *
 * Use a **fine-grained, read-only** token scoped to the one repo.
 */
class Settings(ctx: Context) {

    private val p = ctx.getSharedPreferences("nd", Context.MODE_PRIVATE)

    var baseUrl: String
        get() = p.getString(KEY_URL, null) ?: ModelStore.DEFAULT_BASE_URL
        set(v) = p.edit().putString(KEY_URL, v.trim()).apply()

    var token: String
        get() = p.getString(KEY_TOKEN, null) ?: ""
        set(v) = p.edit().putString(KEY_TOKEN, v.trim()).apply()

    private companion object {
        const val KEY_URL = "base_url"
        const val KEY_TOKEN = "hf_token"
    }
}
