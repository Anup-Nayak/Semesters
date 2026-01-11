WITH procedure_events AS (
    SELECT caregiver_id, COUNT(*) AS procedure_count
    FROM icu.procedureevents 
    GROUP BY caregiver_id
),
chart_events AS (
    SELECT caregiver_id, COUNT(*) AS chart_count
    FROM icu.chartevents
    GROUP BY caregiver_id
),
datetime_events as (
    SELECT caregiver_id, COUNT(*) AS datetime_count
    FROM icu.datetimeevents
    GROUP BY caregiver_id
)

SELECT 
    cg.caregiver_id AS caregiver_id, 
    COALESCE(p.procedure_count, 0) AS procedureevents_count,
    COALESCE(c.chart_count, 0) AS chartevents_count,
    COALESCE(d.datetime_count, 0) AS datetimeevents_count
FROM icu.caregiver cg 
left join procedure_events p 
on p.caregiver_id = cg.caregiver_id
left JOIN chart_events c 
    ON c.caregiver_id = cg.caregiver_id
left JOIN datetime_events d
    ON d.caregiver_id = cg.caregiver_id
    order by caregiver_id, procedureevents_count, chartevents_count, datetimeevents_count;


