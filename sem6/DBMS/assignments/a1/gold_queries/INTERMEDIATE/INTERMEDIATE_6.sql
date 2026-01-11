 with tmp1 as (select subject_id, CAST(result_value AS DECIMAL) as bmi from hosp.omr where result_name='BMI (kg/m2)'),
tmp2 as (select distinct subject_id from hosp.pharmacy where medication ='OxyCODONE (Immediate Release)'),
tmp3 as (select distinct subject_id from hosp.pharmacy where medication ='Insulin'),
tmp4 AS (
    SELECT subject_id 
    FROM tmp2
    INTERSECT
    SELECT subject_id 
    FROM tmp3
)
SELECT  CAST(AVG(tmp1.bmi) AS DECIMAL(20,10)) AS avg_BMI
FROM 
    tmp1 
INNER JOIN 
    tmp4 
ON 
    tmp1.subject_id = tmp4.subject_id;