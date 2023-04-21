variable "public_subnet_cidrs" {
  type        = list(string)
  description = "Public Subnet CIDR values"
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  type        = list(string)
  description = "Private Subnet CIDR values"
  default     = ["10.0.4.0/24", "10.0.5.0/24", "10.0.6.0/24"]
}

variable "azs" {
  type        = list(string)
  description = "Availability Zones"
  default     = ["us-east-2a", "us-east-2b", "us-east-2c"]
}

variable "ami" {
  # Amazon ECS-Optimized Amazon Linux 2 (AL2) x86_64 AMI
  type    = string
  default = "ami-076214eda80ae72ef"
}
variable "keyname" {
  type    = string
  default = "anansi-master-key"
}

variable "application" {
  type    = string
  default = "emebedds"
}

variable "environment" {
  type    = string
  default = "prod"
}

variable "max_instance_count" {
  type    = number
  default = 1
}
variable "min_instance_count" {
  type    = number
  default = 1
}
variable "desired_capacity" {
  type    = number
  default = 1
}

variable "ecs_logging" {
  default     = "[\"json-file\",\"awslogs\"]"
  description = "Adding logging option to ECS that the Docker containers can use. It is possible to add fluentd as well"
}
variable "cloudwatch_prefix" {
  type    = string
  default = "/app"
}
