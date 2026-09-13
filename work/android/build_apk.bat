@echo off
set "JAVA_HOME=C:\Program Files\Android\Android Studio\jbr"
pushd "%~dp0."
call "%~dp0gradlew.bat" %*
popd
