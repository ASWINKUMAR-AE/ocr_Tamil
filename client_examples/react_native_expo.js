/**
 * React Native / Expo Client Integration Example
 * 
 * Works with Expo ImagePicker and native fetch.
 * 
 * IMPORTANT FOR PHYSICAL DEVICES (Expo Go):
 *   - Use your computer's actual LAN IP (e.g., http://10.110.44.105:8000).
 *   - Ensure your phone and development computer are connected to the SAME Wi-Fi network.
 * 
 * IMPORTANT FOR ANDROID EMULATOR:
 *   - If running on an Android Emulator on the same computer, use http://10.0.2.2:8000
 * 
 * IMPORTANT FOR IOS SIMULATOR:
 *   - If running on an iOS Simulator on the same computer, use http://localhost:8000
 */

import React, { useState } from 'react';
import { StyleSheet, Text, View, Button, Image, ScrollView, ActivityIndicator, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

// REPLACE WITH YOUR ACTUAL LAN IP FROM SERVER STARTUP BANNER:
const OCR_SERVER_URL = "http://10.110.44.105:8000";

export default function TamilOCRExpoApp() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [ocrResult, setOcrResult] = useState(null);

  // Pick an image from device gallery
  const pickImage = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert("Permission Required", "Camera roll access is needed to pick an image.");
      return;
    }

    const pickerResult = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 1,
    });

    if (!pickerResult.canceled && pickerResult.assets && pickerResult.assets.length > 0) {
      setSelectedImage(pickerResult.assets[0]);
      setOcrResult(null);
    }
  };

  // Upload image to Tamil OCR REST API
  const performOCR = async () => {
    if (!selectedImage) {
      Alert.alert("No Image", "Please select an image first.");
      return;
    }

    setLoading(true);
    setOcrResult(null);

    try {
      // 1. Prepare FormData
      const formData = new FormData();
      const filename = selectedImage.uri.split('/').pop() || 'upload.jpg';
      const match = /\.(\w+)$/.exec(filename);
      const mimeType = match ? `image/${match[1]}` : 'image/jpeg';

      formData.append('file', {
        uri: selectedImage.uri,
        name: filename,
        type: mimeType,
      });
      formData.append('extract_tables', 'true');
      formData.append('preprocess', 'true');

      // 2. Perform HTTP POST request
      const response = await fetch(`${OCR_SERVER_URL}/api/v1/ocr`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          // If server configured with API_KEY, uncomment:
          // 'X-API-Key': 'YOUR_SECRET_API_KEY',
        },
        body: formData,
      });

      // 3. Handle non-200 responses
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Server responded with HTTP ${response.status}`);
      }

      // 4. Parse JSON result
      const data = await response.json();
      setOcrResult(data);

    } catch (error) {
      Alert.alert("OCR Failed", error.message || "Network error. Check server IP and Wi-Fi connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Tamil OCR Scanner</Text>
      <Text style={styles.subtitle}>Connecting to: {OCR_SERVER_URL}</Text>

      <View style={styles.buttonRow}>
        <Button title="Pick Image" onPress={pickImage} />
        <View style={{ width: 10 }} />
        <Button title="Extract Tamil Text" onPress={performOCR} disabled={!selectedImage || loading} />
      </View>

      {selectedImage && (
        <Image source={{ uri: selectedImage.uri }} style={styles.imagePreview} resizeMode="contain" />
      )}

      {loading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#0066cc" />
          <Text style={styles.loadingText}>Processing Tamil OCR & Tables...</Text>
        </View>
      )}

      {ocrResult && (
        <View style={styles.resultContainer}>
          <Text style={styles.sectionHeader}>Extracted Text:</Text>
          <Text style={styles.tamilText}>{ocrResult.extracted_text || "(No text recognized)"}</Text>

          {ocrResult.tables && ocrResult.tables.length > 0 && (
            <View style={styles.tableContainer}>
              <Text style={styles.sectionHeader}>Detected Tables ({ocrResult.tables.length}):</Text>
              {ocrResult.tables.map((tbl, tIdx) => (
                <View key={tIdx} style={styles.tableCard}>
                  <Text style={styles.tableTitle}>Table #{tbl.table_index + 1} ({tbl.row_count} rows, {tbl.column_count} cols)</Text>
                  <Text style={styles.tableHeaderRow}>Headers: {tbl.headers.join(" | ")}</Text>
                  {tbl.rows.map((row, rIdx) => (
                    <Text key={rIdx} style={styles.tableRow}>Row {rIdx + 1}: {row.join(" | ")}</Text>
                  ))}
                </View>
              ))}
            </View>
          )}

          <Text style={styles.metaText}>
            Engine: {ocrResult.metadata.ocr_engine} | Processing Time: {ocrResult.metadata.processing_time_ms} ms
          </Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, alignItems: 'center', backgroundColor: '#f7f9fa' },
  title: { fontSize: 22, fontWeight: 'bold', color: '#1a1a1a', marginBottom: 4 },
  subtitle: { fontSize: 12, color: '#666', marginBottom: 15 },
  buttonRow: { flexDirection: 'row', marginBottom: 15 },
  imagePreview: { width: '100%', height: 220, borderRadius: 8, marginVertical: 10, backgroundColor: '#eee' },
  loadingContainer: { marginVertical: 20, alignItems: 'center' },
  loadingText: { marginTop: 8, color: '#444' },
  resultContainer: { width: '100%', marginTop: 15, padding: 15, backgroundColor: '#fff', borderRadius: 8, elevation: 2 },
  sectionHeader: { fontSize: 16, fontWeight: 'bold', color: '#0066cc', marginBottom: 6 },
  tamilText: { fontSize: 18, color: '#111', lineHeight: 28, backgroundColor: '#f0f4f8', padding: 10, borderRadius: 6 },
  tableContainer: { marginTop: 15 },
  tableCard: { backgroundColor: '#fefefe', padding: 10, borderRadius: 6, borderWidth: 1, borderColor: '#ddd', marginBottom: 8 },
  tableTitle: { fontWeight: 'bold', fontSize: 14, color: '#333' },
  tableHeaderRow: { fontWeight: '600', color: '#0066cc', marginVertical: 4 },
  tableRow: { fontSize: 13, color: '#444', marginVertical: 2 },
  metaText: { fontSize: 11, color: '#888', marginTop: 10, textAlign: 'right' }
});
