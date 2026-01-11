with diag_set as (
    select subject_id, hadm_id, array_agg(icd_code order by icd_code) as icd_code_agg
    from hosp.diagnoses_icd
    group by subject_id, hadm_id
), medi_set as (
    select subject_id, hadm_id, array_agg(drug order by drug) as drug_agg 
    from hosp.prescriptions
    group by subject_id, hadm_id
), pat_adm as (
    select subject_id, hadm_id
    from hosp.admissions
)
select pa.subject_id, count(distinct pa.hadm_id) as total_admissions, count(distinct COALESCE(icd_code_agg, ARRAY[]::text[])) as num_distinct_diagnoses_set_count, count(distinct COALESCE(drug_agg, ARRAY[]::text[])) as num_distinct_medications_set_count
from pat_adm pa
left join diag_set ds on pa.subject_id = ds.subject_id and pa.hadm_id = ds.hadm_id
left join medi_set ms on pa.subject_id = ms.subject_id and pa.hadm_id = ms.hadm_id
group by pa.subject_id
having count(distinct icd_code_agg) >= 3 or count(distinct drug_agg) >= 3
order by total_admissions desc, num_distinct_diagnoses_set_count desc, subject_id asc;