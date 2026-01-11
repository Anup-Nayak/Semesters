select enter_provider_id, count(distinct medication) 
from hosp.emar 
where enter_provider_id is not NULL 
group by enter_provider_id 
order by count desc;