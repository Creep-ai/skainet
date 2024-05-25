# skainet

#### ИИ ассистент, выполняющий функции DS и DA

#### Запуск
1. Создать `.env` файл и заполнить по аналогии с `.evn.example`
2. Создать папку `postgresql_data`
```shell
mkdir postgresql_data
```
3. Поднять `docker compose`
```shell
sudo docker compose up
```
4. Восстановить БД из дампа
```shell
sudo docker exec -it <db_container_id> bash
pg_restore -U <DB_USER> -d <DB_NAME> /var/dvdrental.tar
```
