# Frontend Update: File Upload Feature

## 🎉 New Feature Added: File Upload

I've enhanced your frontend with a **file upload feature** that allows users to upload JSON or CSV files directly through the web interface. This makes it easy to work with custom datasets.

## 📁 What's New

### Upload Section
- **Drag & Drop Support**: Simply drag files onto the upload area
- **Click to Browse**: Click the upload area to select files
- **Multiple File Support**: Upload multiple JSON/CSV files at once
- **File Management**: View and remove uploaded files

### Supported File Types
- **JSON Files** (.json): Structured data in JSON format
- **CSV Files** (.csv): Comma-separated values, automatically converted to JSON

## 🚀 How to Use

### Starting the System

1. **Start the Backend Server**:
   ```bash
   python start_backend.py
   ```

2. **Open the Frontend**:
   - Navigate to `frontend/index.html`
   - Open it in your web browser

### Uploading Files

1. **Locate the Upload Section**: It's at the top of the main content area
2. **Upload Your Data**:
   - Drag & drop files onto the upload area, OR
   - Click the area to browse and select files
3. **View Uploaded Files**: See the list of files with their types and sizes
4. **Remove Files**: Click the "Remove" button next to any file

### Example File Formats

**JSON Format** (employees.json):
```json
[
  {
    "id": 1,
    "name": "John Doe",
    "department": "Engineering",
    "salary": 85000
  },
  {
    "id": 2,
    "name": "Jane Smith",
    "department": "Marketing",
    "salary": 75000
  }
]
```

**CSV Format** (products.csv):
```csv
product_id,name,price,category
1,Laptop,999.99,Electronics
2,Mouse,29.99,Accessories
3,Keyboard,79.99,Accessories
```

## 🔧 Backend Integration

The uploaded files are:
1. Processed and validated
2. Stored in the `data/` directory
3. Automatically included in the schema context
4. Available for NLP to SQL queries

### New API Endpoint

**POST /upload**
- Accepts multiple files in JSON format
- Processes and stores them for querying
- Returns upload status and record counts

## 📋 Example Queries After Upload

Once you've uploaded custom data, you can query it:
- "Show all products in the Electronics category"
- "List employees with salary above 80000"
- "What's the average price of accessories?"
- "Find the most expensive product"

## 🛠️ Troubleshooting

### Files Not Uploading
- Check file format (must be .json or .csv)
- Ensure the backend server is running
- Check browser console for errors

### CSV Parse Errors
- Ensure CSV has headers in the first row
- Check for proper comma separation
- Avoid special characters in headers

### Query Not Finding Data
- Verify files were uploaded successfully
- Check the filename/table name in your query
- Use the schema endpoint to see available tables

## 🎨 UI Enhancements

The upload interface includes:
- Visual feedback during drag & drop
- File type badges (JSON/CSV)
- File size display
- Smooth animations
- Error handling with user-friendly messages

## 📊 Viewing Uploaded Data

After uploading, you can:
1. Check the stats section for record counts
2. Use the `/schema` endpoint to see table structure
3. Query your data using natural language

## 🚦 Next Steps

With file upload working, you can now:
1. Upload your own datasets
2. Test with real-world data
3. Build complex queries across multiple tables
4. Export query results (future feature)

Your frontend now supports dynamic data loading, making it a complete solution for NLP to SQL conversion with custom datasets!