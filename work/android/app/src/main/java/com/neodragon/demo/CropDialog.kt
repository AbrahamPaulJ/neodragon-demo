package com.neodragon.demo

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Pick the region of a photo that gets animated.
 *
 * The video path is a fixed [targetW] x [targetH], so an arbitrary photo has to lose
 * something. Doing that silently with a centre-crop is the wrong default -- it is the
 * user's picture and the subject is usually not in the middle -- so this lets them place
 * it, and only the chosen region reaches the VAE encoder.
 *
 * The crop is computed in SOURCE pixels rather than from the on-screen preview, so the
 * result does not depend on screen size and loses no resolution to the preview scale.
 *
 * The preview is a Canvas `drawImage` with an explicit destination rect, deliberately
 * NOT an Image with size/scale modifiers. Two earlier attempts got the aspect wrong:
 * `graphicsLayer` scaling of a `ContentScale.Fit` image, and then `Modifier.size()` --
 * which is CLAMPED BY THE PARENT'S CONSTRAINTS, so the moment the zoomed photo grew
 * past the crop window Compose squashed it back to the window. Both made the drawn shape
 * disagree with the crop maths, i.e. what you saw was not what you got.
 *
 * Drawing the rect directly means the preview evaluates the same expressions `crop()`
 * inverts, so the two cannot drift apart.
 */
@Composable
fun CropDialog(
    source: Bitmap,
    targetW: Int,
    targetH: Int,
    onCancel: () -> Unit,
    onDone: (Bitmap) -> Unit,
) {
    Dialog(onDismissRequest = onCancel,
           properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(
            shape = RoundedCornerShape(14.dp),
            color = MaterialTheme.colorScheme.surface,
            modifier = Modifier.fillMaxWidth(0.96f)
        ) {
            Column(Modifier.padding(14.dp)) {
                Text("Choose what to animate",
                     style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(2.dp))
                Text("Everything inside the frame becomes the first video frame " +
                     "(${targetW}x${targetH}). Drag to move, pinch to zoom.",
                     fontSize = 11.sp,
                     color = MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.height(10.dp))

                var scale by remember { mutableFloatStateOf(1f) }
                var offset by remember { mutableStateOf(Offset.Zero) }
                var winW by remember { mutableFloatStateOf(0f) }
                var winH by remember { mutableFloatStateOf(0f) }

                BoxWithConstraints(
                    Modifier.fillMaxWidth()
                        .aspectRatio(targetW.toFloat() / targetH)
                        .clip(RoundedCornerShape(8.dp))
                        .background(Color.Black)
                        // Gestures live on the WINDOW, not the image: with them on the
                        // image, a pinch that started outside the drawn bitmap was
                        // ignored, which reads as "zoom barely responds".
                        .pointerInput(Unit) {
                            detectTransformGestures { _, pan, gz, _ ->
                                // Pinch ratios per event are close to 1.0, so applying
                                // them raw feels dead. Amplifying the delta gives the
                                // travel a photo cropper is expected to have.
                                scale = (scale * (1f + (gz - 1f) * 2.5f)).coerceIn(1f, 12f)
                                offset += pan
                            }
                        }
                ) {
                    val w = constraints.maxWidth.toFloat()
                    val h = constraints.maxHeight.toFloat()
                    winW = w; winH = h

                    // Base scale makes the photo COVER the window, so there is never a
                    // blank edge inside the crop. Zoom multiplies it and never goes below.
                    val base = max(w / source.width, h / source.height)
                    val eff = base * scale
                    val dispW = source.width * eff
                    val dispH = source.height * eff
                    val maxX = max(0f, (dispW - w) / 2f)
                    val maxY = max(0f, (dispH - h) / 2f)
                    offset = Offset(offset.x.coerceIn(-maxX, maxX),
                                    offset.y.coerceIn(-maxY, maxY))

                    // Drawn with an explicit destination rect, NOT with size/scale
                    // modifiers. `Modifier.size()` is clamped by the parent's
                    // constraints, so as soon as the zoomed photo grew past the crop
                    // window Compose squashed it back to the window -- destroying the
                    // aspect ratio and making the preview disagree with the crop.
                    //
                    // `left`/`top`/`dispW`/`dispH` below are the SAME expressions the
                    // crop() function inverts, so preview and result cannot drift apart.
                    val bmp = remember(source) { source.asImageBitmap() }
                    Canvas(Modifier.matchParentSize()) {
                        val left = w / 2f - dispW / 2f + offset.x
                        val top = h / 2f - dispH / 2f + offset.y
                        drawImage(
                            image = bmp,
                            srcOffset = IntOffset.Zero,
                            srcSize = IntSize(source.width, source.height),
                            dstOffset = IntOffset(left.roundToInt(), top.roundToInt()),
                            dstSize = IntSize(dispW.roundToInt(), dispH.roundToInt()),
                        )
                    }

                    // The frame itself, so it is obvious what will be kept.
                    Box(Modifier.matchParentSize()
                        .border(2.dp, Color.White.copy(alpha = 0.85f),
                                RoundedCornerShape(8.dp)))
                }

                Spacer(Modifier.height(6.dp))
                Text("photo ${source.width}x${source.height}    zoom %.1fx".format(scale),
                     fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)

                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    TextButton(onClick = onCancel) { Text("Cancel") }
                    TextButton(onClick = { scale = 1f; offset = Offset.Zero }) {
                        Text("Reset")
                    }
                    Spacer(Modifier.weight(1f))
                    Button(onClick = {
                        onDone(crop(source, targetW, targetH, scale, offset, winW, winH))
                    }) { Text("Use this") }
                }
            }
        }
    }
}

/**
 * Map the on-screen crop window back to source pixels and cut it out.
 *
 * The window spans (0,0)..(winW,winH) in the preview's own coordinates and the photo is
 * drawn centred at scale [eff] with [offset] applied, so its top-left in those same
 * coordinates is `win/2 - disp/2 + offset`. Inverting that gives the source rectangle.
 * Everything is clamped: a rounding error here throws out of createBitmap rather than
 * producing a slightly-off crop.
 */
private fun crop(
    src: Bitmap, targetW: Int, targetH: Int,
    zoom: Float, offset: Offset, winW: Float, winH: Float,
): Bitmap {
    if (winW <= 0f || winH <= 0f) return Bitmap.createScaledBitmap(src, targetW, targetH, true)
    val base = max(winW / src.width, winH / src.height)
    val eff = base * zoom
    val imgLeft = winW / 2f - (src.width * eff) / 2f + offset.x
    val imgTop = winH / 2f - (src.height * eff) / 2f + offset.y

    var sx = ((0f - imgLeft) / eff).roundToInt()
    var sy = ((0f - imgTop) / eff).roundToInt()
    var sw = (winW / eff).roundToInt()
    var sh = (winH / eff).roundToInt()

    sx = sx.coerceIn(0, max(0, src.width - 1))
    sy = sy.coerceIn(0, max(0, src.height - 1))
    sw = sw.coerceIn(1, src.width - sx)
    sh = sh.coerceIn(1, src.height - sy)

    val cut = Bitmap.createBitmap(src, sx, sy, sw, sh)
    // Scale to exactly the model's input size here, so what was approved is what the
    // encoder receives and `bitmapToChw` has nothing left to decide.
    return if (cut.width == targetW && cut.height == targetH) cut
           else Bitmap.createScaledBitmap(cut, targetW, targetH, true)
}
