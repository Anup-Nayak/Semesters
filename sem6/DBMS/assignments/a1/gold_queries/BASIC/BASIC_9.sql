SELECT DISTINCT icd_code, icd_version
FROM hosp.diagnoses_icd
INTERSECT
SELECT DISTINCT icd_code, icd_version
FROM hosp.procedures_icd
ORDER BY icd_code, icd_version;