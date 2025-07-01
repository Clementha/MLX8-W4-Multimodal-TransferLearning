# DuckDB SQL Scripts for Flaticon Dataset Analysis

## Setup
```sql
-- Install and load SQLite extension
INSTALL sqlite;
LOAD sqlite;

-- Attach SQLite database
ATTACH '../.data/flaticon_vision_text.sqlite3' AS sqlite_db (TYPE sqlite);
```

## 0. Sample 10 first images

```sql
SELECT * from sqlite_db.flaticon_images LIMIT 10;

DESCRIBE sqlite_db.flaticon_images;
-- ┌──────────────┬─────────────┬─────────┬─────────┬─────────┬─────────┐
-- │ column_name  │ column_type │  null   │   key   │ default │  extra  │
-- │   varchar    │   varchar   │ varchar │ varchar │ varchar │ varchar │
-- ├──────────────┼─────────────┼─────────┼─────────┼─────────┼─────────┤
-- │ id           │ BIGINT      │ YES     │ PRI     │ NULL    │ NULL    │
-- │ collection   │ VARCHAR     │ NO      │ NULL    │ NULL    │ NULL    │
-- │ type         │ VARCHAR     │ NO      │ NULL    │ NULL    │ NULL    │
-- │ file         │ VARCHAR     │ NO      │ NULL    │ NULL    │ NULL    │
-- │ filename     │ VARCHAR     │ NO      │ NULL    │ NULL    │ NULL    │
-- │ image_path   │ VARCHAR     │ NO      │ NULL    │ NULL    │ NULL    │
-- │ model        │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
-- │ text         │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
-- │ created_when │ TIMESTAMP   │ NO      │ NULL    │ NULL    │ NULL    │
-- │ scanned_when │ TIMESTAMP   │ NO      │ NULL    │ NULL    │ NULL    │
-- │ updated_when │ TIMESTAMP   │ YES     │ NULL    │ NULL    │ NULL    │
-- ├──────────────┴─────────────┴─────────┴─────────┴─────────┴─────────┤
-- │ 11 rows                                                  6 columns │
-- └────────────────────────────────────────────────────────────────────┘
```

## 1. Overall Database Status
```sql
-- Count total records in flaticon_images table
SELECT COUNT(*) as total_images FROM sqlite_db.flaticon_images;

-- Count processed vs unprocessed vs skipped records
SELECT 
    COUNT(*) as total_records,
    COUNT(updated_when) as processed_records,
    SUM(skipped) as skipped_records,
    COUNT(*) - COUNT(updated_when) - SUM(skipped) as unprocessed_records,
    ROUND(COUNT(updated_when) * 100.0 / COUNT(*), 2) as completion_percentage,
    ROUND(SUM(skipped) * 100.0 / COUNT(*), 2) as skipped_percentage
FROM sqlite_db.flaticon_images;
```

## 2. Processing Progress by Collection
```sql
-- Progress breakdown by collection
SELECT 
    collection,
    COUNT(*) as total_images,
    COUNT(updated_when) as processed_images,
    SUM(skipped) as skipped_images,
    COUNT(*) - COUNT(updated_when) - SUM(skipped) as remaining_images,
    ROUND(COUNT(updated_when) * 100.0 / COUNT(*), 2) as completion_pct,
    ROUND(SUM(skipped) * 100.0 / COUNT(*), 2) as skipped_pct
FROM sqlite_db.flaticon_images
GROUP BY collection
ORDER BY completion_pct DESC, total_images DESC;
```

## 3. Processing Progress by File Type
```sql
-- Progress breakdown by file type (png, svg, etc.)
SELECT 
    type,
    COUNT(*) as total_images,
    COUNT(updated_when) as processed_images,
    SUM(skipped) as skipped_images,
    ROUND(COUNT(updated_when) * 100.0 / COUNT(*), 2) as completion_pct,
    ROUND(SUM(skipped) * 100.0 / COUNT(*), 2) as skipped_pct
FROM sqlite_db.flaticon_images
GROUP BY type
ORDER BY completion_pct DESC;
```

## 4. Recent Processing Activity
```sql
-- Last 10 processed records
SELECT 
    id,
    collection,
    file,
    filename,
    updated_when,
    LENGTH(text) as text_length
FROM sqlite_db.flaticon_images
WHERE updated_when IS NOT NULL
ORDER BY updated_when DESC
LIMIT 10;
```

## 5. Processing Count Status
```sql
-- Check last processed record tracking
SELECT 
    image_path as last_processed_image,
    updated_when as last_processing_time
FROM sqlite_db.processing_count
WHERE id = 0;
```

## 6. Error Analysis
```sql
-- Find records that were scanned but never processed (potential errors)
SELECT 
    COUNT(*) as scanned_but_not_processed
FROM sqlite_db.flaticon_images
WHERE scanned_when IS NOT NULL 
AND updated_when IS NULL
AND skipped = 0;

-- Show oldest unprocessed records (excluding skipped)
SELECT 
    id,
    collection,
    file,
    filename,
    created_when,
    scanned_when
FROM sqlite_db.flaticon_images
WHERE updated_when IS NULL AND skipped = 0
ORDER BY created_when ASC
LIMIT 10;
```

## 7. Text Analysis
```sql
-- Average text length by collection
SELECT 
    collection,
    COUNT(text) as processed_count,
    ROUND(AVG(LENGTH(text)), 2) as avg_text_length,
    MIN(LENGTH(text)) as min_text_length,
    MAX(LENGTH(text)) as max_text_length
FROM sqlite_db.flaticon_images
WHERE text IS NOT NULL
GROUP BY collection
ORDER BY avg_text_length DESC;
```

## 8. Daily Processing Stats
```sql
-- Processing activity by date
SELECT 
    DATE(updated_when) as processing_date,
    COUNT(*) as records_processed,
    ROUND(AVG(LENGTH(text)), 2) as avg_text_length
FROM sqlite_db.flaticon_images
WHERE updated_when IS NOT NULL
GROUP BY DATE(updated_when)
ORDER BY processing_date DESC;
```

## 9. Collection Details
```sql
-- Detailed view of a specific collection
SELECT 
    id,
    file,
    filename,
    CASE 
        WHEN updated_when IS NOT NULL THEN 'Processed'
        WHEN scanned_when IS NOT NULL THEN 'Scanned'
        ELSE 'New'
    END as status,
    LENGTH(text) as text_length,
    updated_when
FROM sqlite_db.flaticon_images
WHERE collection = '3720436-smart-home'  -- Replace with actual collection
ORDER BY id;
```

## 10. Export Processed Data
```sql
-- Export processed records to Parquet for analysis
COPY (
    SELECT 
        collection,
        type,
        file,
        filename,
        text,
        model,
        updated_when
    FROM sqlite_db.flaticon_images
    WHERE updated_when IS NOT NULL
) TO '../.data/flaticon_processed_export.parquet' (FORMAT PARQUET);
```

## 11. Skip Management
```sql
-- Mark specific records as skipped
UPDATE sqlite_db.flaticon_images 
SET skipped = 1 
WHERE collection = 'problematic-collection-name';

-- Mark specific file types as skipped
UPDATE sqlite_db.flaticon_images 
SET skipped = 1 
WHERE type = 'svg';

-- Unskip previously skipped records
UPDATE sqlite_db.flaticon_images 
SET skipped = 0 
WHERE collection = 'now-working-collection';

-- View all skipped records
SELECT 
    id,
    collection,
    type,
    file,
    filename,
    created_when
FROM sqlite_db.flaticon_images
WHERE skipped = 1
ORDER BY collection, file;
```

## 12. Skip Analysis
```sql
-- Skipped records by collection
SELECT 
    collection,
    COUNT(*) as total_skipped,
    type
FROM sqlite_db.flaticon_images
WHERE skipped = 1
GROUP BY collection, type
ORDER BY total_skipped DESC;

-- Recently skipped vs processed comparison
SELECT 
    DATE(created_when) as date_added,
    COUNT(*) as total_added,
    SUM(skipped) as skipped_count,
    COUNT(updated_when) as processed_count
FROM sqlite_db.flaticon_images
GROUP BY DATE(created_when)
ORDER BY date_added DESC;
```
