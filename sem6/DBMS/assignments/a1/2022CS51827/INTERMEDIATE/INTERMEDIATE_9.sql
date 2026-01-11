SELECT subject_id, pharmacy_id
FROM hosp.prescriptions
GROUP BY subject_id, pharmacy_id
HAVING COUNT(*) > 1
ORDER BY COUNT(*) DESC, subject_id, pharmacy_id;