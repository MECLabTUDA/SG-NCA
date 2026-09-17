package com.example.sgnca.inference;

import android.content.res.AssetManager;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

final class AssetUtils {
    private AssetUtils() {}

    static byte[] readBytes(AssetManager assets, String path) throws IOException {
        try (InputStream input = assets.open(path);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[16 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
            }
            return output.toByteArray();
        }
    }

    static String readText(AssetManager assets, String path) throws IOException {
        return new String(readBytes(assets, path), StandardCharsets.UTF_8);
    }
}
