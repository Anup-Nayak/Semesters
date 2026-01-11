with multi_proc_admissions as (
    select distinct subject_id
    from hosp.procedures_icd
    group by subject_id, hadm_id
    having count(distinct icd_code)>1
),
get_proc_admns as (
    select f1.subject_id, count(distinct icd_code) as distinct_procedures_count
    from multi_proc_admissions f1
    join hosp.procedures_icd f2 on f1.subject_id = f2.subject_id
    group by f1.subject_id
),
t81_patients as (
    select distinct subject_id
    from hosp.diagnoses_icd 
    where icd_code LIKE'T81%'
),
combined_patients as (
    select f1.subject_id, distinct_procedures_count
    from get_proc_admns f1
    join t81_patients f2 on f1.subject_id = f2.subject_id
),
transfers_sum as (
    select f1.subject_id, f1.distinct_procedures_count,count(*) as number_of_transfers
    from combined_patients f1
    join (select subject_id, hadm_id from hosp.transfers where hadm_id is not NULL)f2 on f1.subject_id = f2.subject_id
    group by f1.subject_id, f1.distinct_procedures_count
),
get_num_admissions as (
    select f1.subject_id, count(*) as num_admissions
    from combined_patients f1
    join hosp.admissions f2 on f1.subject_id = f2.subject_id
    group by f1.subject_id
),
avg_transfer as (
    select f1.subject_id,distinct_procedures_count, number_of_transfers,num_admissions
    from transfers_sum f1
    join get_num_admissions f2 on f1.subject_id = f2.subject_id
),
overall_avg_transfer as (
    select sum(number_of_transfers)/sum(num_admissions) as overall_avg
    from avg_transfer
)

select a.subject_id,a.distinct_procedures_count, round(a.number_of_transfers/a.num_admissions::DECIMAL , 2)  as average_transfers
from avg_transfer a
join overall_avg_transfer o ON a.number_of_transfers/a.num_admissions::DECIMAL >= o.overall_avg
order by 
    average_transfers DESC, 
    distinct_procedures_count DESC,  
    subject_id ASC;