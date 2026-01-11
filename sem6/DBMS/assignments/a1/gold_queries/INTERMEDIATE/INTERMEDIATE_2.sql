SELECT AVG(duration) as avg_duration from
    (select a.hadm_id, a.dischtime::TIMESTAMP - a.admittime::TIMESTAMP as duration, d.icd_code as icd_code, d.icd_version as icd_version  
    from hosp.admissions a join hosp.diagnoses_icd d on a.subject_id=d.subject_id and a.hadm_id=d.hadm_id 
    where d.icd_code='4019' and d.icd_version='9' and a.dischtime is not NULL and a.admittime is not NULL) 
AS subquery;