select count(distinct subject_id), DATE_PART('year', TO_TIMESTAMP(admittime, 'YYYY-MM-DD HH24:MI:SS'))::INTEGER as year  
from hosp.admissions 
group by DATE_PART('year', TO_TIMESTAMP(admittime, 'YYYY-MM-DD HH24:MI:SS')) 
order by count(distinct subject_id) desc, year limit 5;