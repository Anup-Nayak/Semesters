select a.subject_id, b.hadm_id, a.dod 
from hosp.patients a join hosp.admissions b on a.subject_id=b.subject_id  
where dod is not NULL 
and b.admittime=(select min(sub.admittime) from hosp.admissions sub where sub.subject_id=b.subject_id)
order by a.subject_id;