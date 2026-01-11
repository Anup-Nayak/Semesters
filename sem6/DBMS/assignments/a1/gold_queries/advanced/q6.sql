WITH fluid_input AS (
    SELECT
        stay_id,
        SUM(amount) AS total_input
    FROM icu.inputevents
    WHERE LOWER(amountuom) = 'ml'  -- Consider only amounts recorded in milliliters
    GROUP BY stay_id
),

fluid_output AS (
    SELECT
        stay_id,
        SUM(value) AS total_output
    FROM icu.outputevents
    WHERE LOWER(valueuom) = 'ml'  -- Consider only amounts recorded in milliliters
    GROUP BY stay_id
),

fluid_balance AS (
    SELECT
        COALESCE(i.stay_id, o.stay_id) AS stay_id,
        COALESCE(i.total_input, 0) AS total_input_ml,
        COALESCE(o.total_output, 0) AS total_output_ml,
        COALESCE(i.total_input, 0) - COALESCE(o.total_output, 0) AS net_fluid_balance_ml
    FROM fluid_input i
    FULL OUTER JOIN fluid_output o
        ON i.stay_id = o.stay_id
    WHERE ABS(COALESCE(i.total_input, 0) - COALESCE(o.total_output, 0)) > 2000  -- Exceeds 2000 mL
),

patient_info AS (
    SELECT DISTINCT
        fb.stay_id,
        p.subject_id
    FROM fluid_balance fb
    JOIN icu.icustays p ON fb.stay_id = p.stay_id
),

all_events AS (
    SELECT stay_id, itemid, 'input' AS input_or_output FROM icu.inputevents
    UNION
    SELECT stay_id, itemid, 'output' AS input_or_output FROM icu.outputevents

)

SELECT distinct
    pi.subject_id,
    fb.stay_id,
    ae.itemid,
    ae.input_or_output,
    d.abbreviation AS description
FROM fluid_balance fb
JOIN patient_info pi ON fb.stay_id = pi.stay_id
JOIN all_events ae ON fb.stay_id = ae.stay_id
JOIN icu.d_items d ON ae.itemid = d.itemid
ORDER BY pi.subject_id, fb.stay_id, ae.itemid, ae.input_or_output;
