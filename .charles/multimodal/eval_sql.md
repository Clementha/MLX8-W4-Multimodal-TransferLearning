Check results:


```sql
duckdb -c "SELECT * FROM '../.data/models/04_vit_top0_flickr30k_test_results.parquet' LIMIT 10;"

duckdb -c "SELECT reference, prediction FROM '../.data/models/04_vit_top0_flickr30k_test_results.parquet' LIMIT 50;"
```
