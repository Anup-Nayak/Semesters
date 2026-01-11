SELECT COUNT(DISTINCT e.hadm_id) AS count
FROM hosp.emar_detail ed
JOIN hosp.emar e ON ed.emar_id = e.emar_id
JOIN hosp.admissions a ON e.hadm_id = a.hadm_id
WHERE ed.reason_for_no_barcode = 'Barcode Damaged'
AND a.marital_status <> 'MARRIED'
AND a.marital_status IS NOT NULL;