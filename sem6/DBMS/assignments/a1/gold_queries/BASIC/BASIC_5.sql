SELECT COUNT(DISTINCT a.hadm_id)
FROM hosp.admissions a
JOIN hosp.emar e ON a.subject_id = e.subject_id 
                 AND a.hadm_id = e.hadm_id
JOIN hosp.emar_detail ed ON e.emar_id = ed.emar_id
WHERE ed.reason_for_no_barcode = 'Barcode Damaged' 
  AND a.marital_status != 'MARRIED';