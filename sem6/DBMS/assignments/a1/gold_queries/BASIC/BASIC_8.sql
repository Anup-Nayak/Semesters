SELECT pharmacy_id
FROM hosp.pharmacy
EXCEPT
SELECT distinct pharmacy_id
FROM hosp.prescriptions
order by pharmacy_id;