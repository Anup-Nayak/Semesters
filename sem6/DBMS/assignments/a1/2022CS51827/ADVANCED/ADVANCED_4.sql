WITH TransferChains AS (
    SELECT hadm_id, subject_id, 
           array_agg(transfer_id ORDER BY intime) AS transfers,
           COUNT(*) AS chain_length
    FROM hosp.transfers
    WHERE hadm_id IS NOT NULL
    GROUP BY hadm_id, subject_id
),
MaxChain AS (
    SELECT chain_length
    FROM TransferChains
    ORDER BY chain_length DESC
    LIMIT 1
),
MaxChainAdmissions AS (
    SELECT hadm_id, subject_id
    FROM TransferChains
    WHERE chain_length = (SELECT chain_length FROM MaxChain)
),
RelevantAdmissions AS (
    SELECT DISTINCT t.hadm_id, t.subject_id
    FROM MaxChainAdmissions m
    JOIN hosp.transfers t ON m.subject_id = t.subject_id
    WHERE t.hadm_id IS NOT NULL
)
SELECT r.subject_id, r.hadm_id, 
       COALESCE(array_to_string(tc.transfers, ','), '') AS "[transfers]"
FROM RelevantAdmissions r
LEFT JOIN TransferChains tc ON r.hadm_id = tc.hadm_id
WHERE r.hadm_id IS NOT NULL
ORDER BY array_length(transfers, 1), r.hadm_id, r.subject_id;