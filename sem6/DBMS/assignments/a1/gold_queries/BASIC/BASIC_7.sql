select a.subject_id, max(b.hadm_id) as latest_hadm_id, a.dod 
from hosp.patients a join hosp.admissions b 
on a.subject_id=b.subject_id  
where dod is not NULL 
group by a.subject_id, a.dod
 order by a.subject_id;