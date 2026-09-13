plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.neodragon.demo"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.neodragon.demo"
        minSdk = 31
        targetSdk = 35
        versionCode = 2
        versionName = "0.2.0"
        // filename carries the version: several builds end up in the
        // phone's Downloads folder and "app-debug.apk" tells you nothing.
        setProperty("archivesBaseName", "neodragon-demo-$versionName")
        ndk { abiFilters += "arm64-v8a" }

        // Target Hexagon revision, and where the binaries compiled for it are downloaded
        // from. Defaults are the S25 Ultra build. For another SoC, convert the models with
        // HTP_ARCH/SOC_MODEL set and build with e.g.
        //   ./gradlew assembleDebug -PhtpArch=v75 -PsocLabel="Snapdragon 8 Gen 3 / SM8650"         //       -PmodelBaseUrl=https://huggingface.co/<you>/<repo>/resolve/main
        // See docs/porting-other-socs.md.
        fun prop(k: String, d: String) = (project.findProperty(k) as String?) ?: d
        buildConfigField("String", "HTP_ARCH", "\"${prop("htpArch", "v79")}\"")
        buildConfigField("String", "SOC_LABEL", "\"${prop("socLabel", "Snapdragon 8 Elite / SM8750")}\"")
        buildConfigField("String", "MODEL_BASE_URL",
            "\"${prop("modelBaseUrl", "https://huggingface.co/AbrahamPJ/neodragon-npu-s25u/resolve/main")}\"")
        externalNativeBuild {
            cmake { arguments += listOf("-DANDROID_STL=c++_shared") }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    ndkVersion = "27.2.12479018"

    // The QNN runtime .so files ship in jniLibs and are dlopen'd by libndqnn.so at
    // runtime. They are NOT exec'd: a process exec'd out of the APK is denied the
    // Hexagon fastrpc device (see the header comment in ndqnn.cpp -- the identical
    // binary works from /data/local/tmp and fails from /data/app), so the backend has
    // to be loaded into the app process itself. useLegacyPackaging keeps the .so files
    // as real files on disk, which dlopen by absolute path requires.
    packaging {
        jniLibs {
            useLegacyPackaging = true
            keepDebugSymbols += "**/*.so"
        }
    }

    buildTypes {
        debug { isMinifyEnabled = false }
        release { isMinifyEnabled = false }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true; buildConfig = true }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation(platform("androidx.compose:compose-bom:2024.10.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
}
