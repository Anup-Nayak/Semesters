with first_adm as (
    select subject_id, MIN(admittime) as min_admittime
    from hosp.admissions
    group by subject_id
), kidney_pat as (
    select distinct adm.subject_id, adm.admittime
    from hosp.admissions adm
    join first_adm on adm.subject_id = first_adm.subject_id and adm.admittime = first_adm.min_admittime
     join hosp.diagnoses_icd diag on adm.hadm_id = diag.hadm_id
     join hosp.d_icd_diagnoses d_icd on diag.icd_code = d_icd.icd_code and diag.icd_version = d_icd.icd_version
    where lower(long_title) LIKE '%kidney%'
), tmp as (select distinct a.subject_id, kidney_pat.admittime from hosp.admissions a join kidney_pat on a.subject_id=kidney_pat.subject_id and a.admittime>kidney_pat.admittime order by kidney_pat.admittime desc limit 100)
select subject_id from tmp order by subject_id ;