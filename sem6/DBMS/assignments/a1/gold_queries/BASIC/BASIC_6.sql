select a.subject_id, count(distinct i.stay_id) 
from hosp.admissions a left JOIN icu.icustays i 
on a.subject_id=i.subject_id 
group by a.subject_id
order by count, subject_id;