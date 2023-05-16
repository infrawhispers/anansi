
resource "aws_security_group" "embeddings-api-sg" {
  name        = "emebddings-api-sg"
  description = "allow inbound traffic to the embedding api for HTTPS + gRPC from alb"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "HTTPS traffic into the alb"
    from_port       = 50051
    to_port         = 50051
    protocol        = "tcp"
    security_groups = [aws_security_group.embeddings-lb-sg.id]
  }
  ingress {
    description     = "HTTPS traffic into the alb"
    from_port       = 50052
    to_port         = 50052
    protocol        = "tcp"
    security_groups = [aws_security_group.embeddings-lb-sg.id]
  }

  ingress {
    description = ""
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["24.90.76.202/32"]
  }
  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}


resource "aws_security_group" "embeddings-lb-sg" {
  name        = "embeddings-lb-sg"
  description = "allow inbound traffic to the embeddings api for HTTPS + gRPC"
  vpc_id      = aws_vpc.main.id

  ingress {
    description      = "gRPC traffic into the alb"
    from_port        = 50051
    to_port          = 50051
    protocol         = "tcp"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
  ingress {
    description      = "HTTPS traffic into the alb"
    from_port        = 50052
    to_port          = 50052
    protocol         = "tcp"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}

resource "aws_alb" "embeddings-lb" {
  name               = "embeddings-lb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.embeddings-lb-sg.id]
  subnets            = [for subnet in aws_subnet.public_subnets : subnet.id]
  enable_http2       = true
}

# resource "aws_alb" "embeddings-lb-http" {
#   name               = "embeddings-lb-http"
#   internal           = false
#   load_balancer_type = "application"
# }

resource "aws_lb_listener" "embeddings-lb-grpc" {
  load_balancer_arn = aws_alb.embeddings-lb.arn
  port              = "50051"
  protocol          = "HTTPS"
  certificate_arn   = aws_acm_certificate.embeddings-cert.arn
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.embeddings-lb-tg-grpc.id
  }
}

resource "aws_lb_listener" "embeddings-lb-http" {
  load_balancer_arn = aws_alb.embeddings-lb.arn
  port              = "50052"
  protocol          = "HTTPS"
  certificate_arn   = aws_acm_certificate.embeddings-cert.arn
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.embeddings-lb-tg-http.id
  }
}

resource "aws_lb_target_group" "embeddings-lb-tg-grpc" {
  name             = "embeddings-lb-tg-grpc"
  port             = 50051
  protocol         = "HTTP"
  protocol_version = "GRPC"
  target_type      = "instance"
  vpc_id           = aws_vpc.main.id
  health_check {
    enabled             = true
    matcher             = "0"
    path                = "/grpc.health.v1.Health/Check"
    port                = 50051
    protocol            = "HTTP"
    unhealthy_threshold = 5
  }
}
resource "aws_lb_target_group" "embeddings-lb-tg-http" {
  name             = "embeddings-lb-tg-http"
  port             = 50052
  protocol         = "HTTP"
  protocol_version = "HTTP1"
  target_type      = "instance"
  vpc_id           = aws_vpc.main.id
  health_check {
    enabled             = true
    matcher             = "200"
    path                = "/health"
    port                = 50053
    protocol            = "HTTP"
    unhealthy_threshold = 5
  }
}


resource "aws_route53_record" "embeddings_primary" {
  zone_id = "Z02443361LULRKGC1XDNR"
  name    = "api.embeddings"
  type    = "A"
  alias {
    name                   = aws_alb.embeddings-lb.dns_name
    zone_id                = aws_alb.embeddings-lb.zone_id
    evaluate_target_health = true
  }
}


resource "aws_efs_file_system" "backend-efs" {
  creation_token = "backend-efs"
}
resource "aws_security_group" "backend-efs-sg" {
  name        = "backend-efs-sg"
  description = "allow inbound and outbound traffic to access EFS"
  vpc_id      = aws_vpc.main.id
  ingress {
    description     = "efs network traffic into and our of the mount point"
    from_port       = 0
    to_port         = 0
    protocol        = "-1"
    security_groups = [aws_security_group.embeddings-api-sg.id]
  }
  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}
resource "aws_efs_mount_target" "backend-efs-mnt-1" {
  file_system_id  = aws_efs_file_system.backend-efs.id
  subnet_id       = element(aws_subnet.public_subnets, 0).id
  security_groups = [aws_security_group.backend-efs-sg.id]
}
resource "aws_efs_mount_target" "backend-efs-mnt-2" {
  file_system_id  = aws_efs_file_system.backend-efs.id
  subnet_id       = element(aws_subnet.public_subnets, 1).id
  security_groups = [aws_security_group.backend-efs-sg.id]
}
resource "aws_efs_mount_target" "backend-efs-mnt-3" {
  file_system_id  = aws_efs_file_system.backend-efs.id
  subnet_id       = element(aws_subnet.public_subnets, 2).id
  security_groups = [aws_security_group.backend-efs-sg.id]
}
data "aws_iam_policy_document" "backend-efs-policy" {
  statement {
    sid    = "BackendEFSAllowIAM"
    effect = "Allow"
    principals {
      type        = "AWS"
      identifiers = ["*"]
    }
    actions = [
      "elasticfilesystem:ClientMount",
      "elasticfilesystem:ClientWrite",
    ]
    resources = [aws_efs_file_system.backend-efs.arn]
  }
  statement {
    sid    = "BackendEFSAllowIAM-ECSTask"
    effect = "Allow"
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::151946447080:role/prod_ecs_instance_role"]
    }
    actions = [
      "elasticfilesystem:ClientMount",
      "elasticfilesystem:ClientWrite",
      "elasticfilesystem:DescribeMountTargets",
      "elasticfilesystem:ClientRootAccess",
    ]
    resources = [aws_efs_file_system.backend-efs.arn]
  }
}
resource "aws_efs_file_system_policy" "backend-efs-system-policy" {
  file_system_id                     = aws_efs_file_system.backend-efs.id
  bypass_policy_lockout_safety_check = false
  policy                             = data.aws_iam_policy_document.backend-efs-policy.json
}

### now build the things necessary to run ECS on our containers.
resource "aws_launch_configuration" "backend" {
  name_prefix          = "backend-launch-config"
  image_id             = var.ami
  instance_type        = "c5.xlarge"
  key_name             = var.keyname
  security_groups      = [aws_security_group.embeddings-api-sg.id]
  user_data            = data.template_file.user_data.rendered
  iam_instance_profile = aws_iam_instance_profile.ecs.id
  root_block_device {
    volume_type = "gp2"
    volume_size = 48
    encrypted   = true
  }
}

resource "aws_autoscaling_group" "backend-asg" {
  name = "${var.application}-ecs-${var.environment}"
  vpc_zone_identifier = [
    element(aws_subnet.public_subnets, 0).id,
    element(aws_subnet.public_subnets, 1).id,
    element(aws_subnet.public_subnets, 2).id
  ]
  max_size                  = var.max_instance_count
  min_size                  = var.min_instance_count
  health_check_grace_period = 300
  health_check_type         = "ELB"
  desired_capacity          = var.desired_capacity
  force_delete              = true
  launch_configuration      = aws_launch_configuration.backend.name

  depends_on            = [aws_launch_configuration.backend]
  protect_from_scale_in = true

  tag {
    key                 = "AmazonECSManaged"
    value               = true
    propagate_at_launch = true
  }
}

resource "aws_ecs_capacity_provider" "backend-ecs-cp" {
  name = "backend-ecs-cp"
  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.backend-asg.arn
    managed_termination_protection = "ENABLED"

    managed_scaling {
      maximum_scaling_step_size = 1000
      minimum_scaling_step_size = 1
      status                    = "ENABLED"
      target_capacity           = 1
    }
  }
}
resource "aws_ecs_cluster" "backend-ecs-cluster" {
  name = "backend-ecs-cluster"
}

resource "aws_ecs_cluster_capacity_providers" "example" {
  cluster_name       = aws_ecs_cluster.backend-ecs-cluster.name
  capacity_providers = [aws_ecs_capacity_provider.backend-ecs-cp.name]
  default_capacity_provider_strategy {
    base              = 1
    weight            = 100
    capacity_provider = aws_ecs_capacity_provider.backend-ecs-cp.name
  }
}

data "template_file" "user_data" {
  template = file("${path.module}/templates/user_data.sh")

  vars = {
    ecs_config        = ""
    ecs_logging       = var.ecs_logging
    cluster_name      = aws_ecs_cluster.backend-ecs-cluster.name
    env_name          = var.environment
    custom_userdata   = ""
    cloudwatch_prefix = var.cloudwatch_prefix
  }
}
