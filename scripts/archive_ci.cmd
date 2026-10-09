@echo off
rem Weekly incremental mirror of every repository under data\ (archives CI logs before GitHub deletes them).
rem Remove with:  schtasks /Delete /TN ospm-ci-archive /F
cd /d D:\capstone
"C:\Python312\python.exe" -m ospm.cli archive --data-root data >> "D:\capstone\data\archive_stdout.log" 2>&1
