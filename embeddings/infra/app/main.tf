resource "aws_ecs_service" "embedds-app" {
  name                    = "embedds-app"
  cluster                 = "backend-ecs-cluster"
  task_definition         = aws_ecs_task_definition.embedds-task.arn
  desired_count           = 1
  enable_ecs_managed_tags = true
  # iam_role        = aws_iam_role.foo.arn
  # depends_on      = [aws_iam_role_policy.foo]
  ordered_placement_strategy {
    type  = "binpack"
    field = "cpu"
  }
  load_balancer {
    target_group_arn = "arn:aws:elasticloadbalancing:us-east-2:151946447080:targetgroup/embeddings-lb-tg-grpc/ffc99a0bcf98fa29"
    container_name   = "app"
    container_port   = 50051
  }
  # load_balancer {
  #   target_group_arn = "arn:aws:elasticloadbalancing:us-east-2:151946447080:targetgroup/embeddings-lb-tg-grpc/6a1f58a7a9eb953b"
  #   container_name   = "app"
  #   container_port   = 50052
  # }
  placement_constraints {
    type       = "memberOf"
    expression = "attribute:ecs.availability-zone in [us-east-2a, us-east-2b, us-east-2c]"
  }
  # use this if we figure out how to get awsvpc working correctly - will likely need
  # a NAT since we rely on the Transformers module and do not cache things. 
  # network_configuration {
  #   subnets = [
  #     "subnet-0c1668f3fdbf4c1ef", "subnet-07020e2f325aceb64", "subnet-01ac19546478793c9"
  #   ]
  #   security_groups = ["sg-04a1b50c33d4ed86b"]
  # }
}

resource "aws_ecs_task_definition" "embedds-task" {
  family             = "embedds"
  task_role_arn      = "arn:aws:iam::151946447080:role/prod-execution-task-role"
  execution_role_arn = "arn:aws:iam::151946447080:role/prod-execution-task-role"
  network_mode       = "host"
  container_definitions = jsonencode([
    {
      "name" : "app",
      "image" : "151946447080.dkr.ecr.us-east-2.amazonaws.com/anansi.embeddings:latest-cpu",
      "memory" : 4000,
      "cpu" : 2048,
      "command" : ["ls", "-lah", "/app/.cache/cache"],
      "environment" : [
        {
          "name" : "EMBEDDS_GRPC_PORT",
          "value" : "50051"
        },
        {
          "name" : "EMBEDDS_HTTP_PORT",
          "value" : "50052"
        },
        {
          "name" : "EMBEDDS_CACHE_FOLDER",
          "value" : "/app/.cache/cache"
        }
      ],
      "portMappings" : [
        {
          "containerPort" : 50051,
          "protocol" : "tcp"
        },
        {
          "containerPort" : 50052,
          "protocol" : "tcp"
        }
      ],
      "mountPoints" : [
        {
          "sourceVolume" : "model-cache",
          "containerPath" : "/app/.cache",
          "readOnly" : true
        }
      ],
      "logConfiguration" : {
        "logDriver" : "awslogs",
        "options" : {
          "awslogs-group" : "nginx-container",
          "awslogs-region" : "us-east-2",
          "awslogs-create-group" : "true",
          "awslogs-stream-prefix" : "nginx"
        }
      }
    }
  ])
  volume {
    name = "model-cache"
    efs_volume_configuration {
      file_system_id     = "fs-070700d63400179c1"
      root_directory     = "cache/"
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = "fsap-0244f8223d311d260"
        iam             = "ENABLED"
      }
    }
  }
}
