WITH FluidBalance AS (
    SELECT ie.subject_id, ie.stay_id,
           COALESCE(SUM(ie.amount), 0) AS total_input,
           COALESCE(SUM(oe.value), 0) AS total_output
    FROM icu.inputevents ie
    LEFT JOIN icu.outputevents oe ON ie.stay_id = oe.stay_id
    WHERE ie.amountuom = 'ml' AND oe.valueuom = 'ml'
    GROUP BY ie.subject_id, ie.stay_id
    HAVING ABS(COALESCE(SUM(ie.amount), 0) - COALESCE(SUM(oe.value), 0)) > 2000
),
DistinctItems AS (
    SELECT fb.subject_id, fb.stay_id, ie.itemid AS item_id, 'input' AS input_or_output
    FROM FluidBalance fb
    JOIN icu.inputevents ie ON fb.stay_id = ie.stay_id
    WHERE ie.amountuom = 'ml'
    
    UNION
    
    SELECT fb.subject_id, fb.stay_id, oe.itemid AS item_id, 'output' AS input_or_output
    FROM FluidBalance fb
    JOIN icu.outputevents oe ON fb.stay_id = oe.stay_id
    WHERE oe.valueuom = 'ml'
)
SELECT di.subject_id, di.stay_id, di.item_id, di.input_or_output, d_items.label AS description
FROM DistinctItems di
JOIN icu.d_items d_items ON di.item_id = d_items.itemid
ORDER BY di.subject_id ASC, di.stay_id ASC, di.item_id ASC, di.input_or_output ASC;