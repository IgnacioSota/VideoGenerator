# VideoGenerator



Este Trabajo de Fin de Grado presenta el diseño e implementación de un sistema en Python capaz de generar automáticamente un montaje audiovisual a partir de uno o varios vídeos de entrada y una pista de audio seleccionada por el usuario. Estas son las instrucciones y comandos que hay que utilizar para poder usar el programa. Se ha utilizado Anaconda Prompt para la ejecución de comandos.





### Creación de Entorno (En terminal):



&nbsp;	python -m venv TFGEnv

&nbsp;	.\\TFGEnv\\Scripts\\activate	





### Obtención de los vídeos de entrenamiento: 

&nbsp;	python get\_training\_videos/download\_training\_videos.py



### Creación y entrenamiento del modelo: 

&nbsp;	python ml\_model/generate\_training\_data.py



### Ejecución del programa principal: 

&nbsp;	python main.py



### Evaluación del la importancia de las variables del modelo:

&nbsp;	python analyze\_feature\_importance.py



