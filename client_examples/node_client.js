/**
 * Node.js Client Example for Tamil OCR REST API
 * Compatible with Node.js 18+ (uses native fetch, FormData, and Blob).
 * 
 * Usage:
 *   node node_client.js <path-to-image> [server-url] [api-key]
 * Example:
 *   node node_client.js sample_images/0.jpg http://10.110.44.105:8000
 */

const fs = require('fs');
const path = require('path');

const imagePath = process.argv[2] || 'sample_images/0.jpg';
const serverUrl = (process.argv[3] || 'http://10.110.44.105:8000').replace(/\/+$/, '');
const apiKey = process.argv[4] || '';

async function runNodeOCR() {
  if (!fs.existsSync(imagePath)) {
    console.error(`Error: File '${imagePath}' not found.`);
    process.exit(1);
  }

  console.log(`Sending image to Tamil OCR API at: ${serverUrl}/api/v1/ocr`);
  console.log(`Image: ${imagePath}`);

  try {
    const fileBuffer = fs.readFileSync(imagePath);
    const fileName = path.basename(imagePath);
    const blob = new Blob([fileBuffer], { type: 'image/png' });

    const formData = new FormData();
    formData.append('file', blob, fileName);
    formData.append('extract_tables', 'true');
    formData.append('preprocess', 'true');

    const headers = { 'Accept': 'application/json' };
    if (apiKey) {
      headers['X-API-Key'] = apiKey;
    }

    const startTime = Date.now();
    const response = await fetch(`${serverUrl}/api/v1/ocr`, {
      method: 'POST',
      headers: headers,
      body: formData
    });

    const elapsed = Date.now() - startTime;

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      console.error(`HTTP ${response.status} Error:`, JSON.stringify(err, null, 2));
      process.exit(1);
    }

    const data = await response.json();

    console.log('\n========================================');
    console.log('         OCR EXTRACTION SUCCESS         ');
    console.log('========================================');
    console.log(`Request ID:      ${data.request_id}`);
    console.log(`File:            ${data.filename}`);
    console.log(`Processing Time: ${data.metadata.processing_time_ms} ms (Total roundtrip: ${elapsed} ms)`);
    console.log('----------------------------------------');
    console.log('EXTRACTED TAMIL TEXT:');
    console.log('----------------------------------------');
    console.log(data.extracted_text || '(No text detected)');
    console.log('----------------------------------------');

    if (data.tables && data.tables.length > 0) {
      console.log(`DETECTED TABLES (${data.tables.length}):`);
      data.tables.forEach((tbl, i) => {
        console.log(`\nTable #${i + 1} (${tbl.row_count} rows x ${tbl.column_count} cols)`);
        console.log('Headers: ', tbl.headers.join(' | '));
        tbl.rows.forEach((r, rIdx) => {
          console.log(`Row ${rIdx + 1}:  ${r.join(' | ')}`);
        });
      });
    }

    if (data.warnings && data.warnings.length > 0) {
      console.log('\nWarnings:', data.warnings);
    }

  } catch (error) {
    console.error('Network or Execution Error:', error.message);
    console.error('Make sure the server is reachable and running on the LAN IP and port.');
    process.exit(1);
  }
}

runNodeOCR();
