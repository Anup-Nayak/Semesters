SELECT p.pharmacy_id
FROM hosp.pharmacy p
LEFT JOIN hosp.prescriptions pr ON p.pharmacy_id = pr.pharmacy_id
WHERE pr.pharmacy_id IS NULL
ORDER BY p.pharmacy_id;