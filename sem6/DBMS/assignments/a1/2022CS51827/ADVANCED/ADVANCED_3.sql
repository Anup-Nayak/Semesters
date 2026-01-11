WITH procedure_counts AS (
    SELECT subject_id, hadm_id, COUNT(DISTINCT icd_code) AS distinct_procedures_count
    FROM hosp.procedures_icd
    WHERE hadm_id IS NOT NULL
    GROUP BY subject_id, hadm_id
    HAVING COUNT(DISTINCT icd_code) > 1
),
diagnosed_patients AS (
    SELECT DISTINCT subject_id, hadm_id
    FROM hosp.diagnoses_icd
    WHERE icd_code LIKE 'T81%' AND hadm_id IS NOT NULL
),
transfer_counts AS (
    SELECT subject_id, hadm_id, COUNT(*) AS num_transfers
    FROM hosp.transfers
    WHERE hadm_id IS NOT NULL
    GROUP BY subject_id, hadm_id
),
avg_transfers AS (
    SELECT AVG(num_transfers) AS avg_transfers_per_admission FROM transfer_counts
)
SELECT p.subject_id, p.distinct_procedures_count,
    CAST(t.num_transfers AS DOUBLE PRECISION) AS average_transfers
FROM procedure_counts p
JOIN diagnosed_patients d ON p.subject_id = d.subject_id AND p.hadm_id = d.hadm_id
JOIN transfer_counts t ON p.subject_id = t.subject_id AND p.hadm_id = t.hadm_id
WHERE t.num_transfers >= (SELECT avg_transfers_per_admission FROM avg_transfers)
ORDER BY average_transfers DESC, p.distinct_procedures_count DESC, p.subject_id;