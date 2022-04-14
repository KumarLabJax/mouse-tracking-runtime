This directory contains files for building the docker image used by the Google pose estimation pipeline.
The cloudbuild.yaml file can be used to submit the build to Google Cloud Build. The cloudbuild.yaml file 
currently specifies the jax-jmcrs-behavior-sb-01 project -- you'll need appropriate permissions for this 
project to submit the cloud build job and to push the resulting docker image to the project's 
container repository.

To submit the cloud build job, run the following command from the *project root directory* (up one 
level from this directory). 

```shell
gcloud builds submit --config docker/cloudbuild.yaml --timeout=30m
```

Note, this image is based on a Google GPU enabled PyTorch 1.7 image. This is a large image with a lot
of layers, and can take a long time to pull. The default cloud build timeout is not always long 
enough for the build to complete sucessfully, so you may need to increase it using the --timeout
argument. A timeout of "30m" (30 minutes) should be more than sufficient. 
