# skainet

#### ИИ ассистент, выполняющий функции DS и DA

#### Запуск
```shell
mkdir postgresql_data
sudo docker compose up
sudo docker exec -it <db_container_id> bash
pg_restore -U <DB_USER> -d <DB_NAME> /var/dvdrental.tar
```
