SELECT CAST(AVG(CAST(result_value AS DOUBLE PRECISION)) AS DECIMAL(12,10)) AS avg_BMI
FROM hosp.omr
WHERE result_name = 'BMI (kg/m2)' 
AND subject_id IN (
    SELECT DISTINCT subject_id
    FROM hosp.prescriptions
    WHERE drug = 'Insulin'
    INTERSECT
    SELECT subject_id
    FROM hosp.prescriptions
    WHERE drug = 'OxyCODONE (Immediate Release)'
);