SELECT enter_provider_id, COUNT(DISTINCT medication) AS count
FROM hosp.emar
WHERE enter_provider_id IS NOT NULL
GROUP BY enter_provider_id
ORDER BY count DESC;