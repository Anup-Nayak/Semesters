with tmp1 as 
    (select subject_id, pharmacy_id, count(hadm_id) 
    from hosp.prescriptions 
    group by subject_id, pharmacy_id  
    having count(hadm_id)>1 )
select subject_id, pharmacy_id 
from tmp1 
order by count desc, subject_id, pharmacy_id;