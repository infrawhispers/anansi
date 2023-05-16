resource "aws_ecr_repository" "ecr_primary" {
  name                 = "anansi.embeddings"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  tags = {
    Name = "anansi primary VPC"
  }
}

resource "aws_internet_gateway" "main-ig" {
  vpc_id = aws_vpc.main.id
}

resource "aws_subnet" "public_subnets" {
  count                   = length(var.public_subnet_cidrs)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = element(var.public_subnet_cidrs, count.index)
  availability_zone       = element(var.azs, count.index)
  map_public_ip_on_launch = true
  tags = {
    Name = "Public Subnet ${count.index + 1}"
  }
}

resource "aws_subnet" "private_subnets" {
  count                   = length(var.private_subnet_cidrs)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = element(var.private_subnet_cidrs, count.index)
  availability_zone       = element(var.azs, count.index)
  map_public_ip_on_launch = false
  tags = {
    Name = "Private Subnet ${count.index + 1}"
  }
}

resource "aws_acm_certificate" "embeddings-cert" {
  domain_name       = "api.embeddings.getanansi.com"
  validation_method = "DNS"

}

resource "aws_route_table" "pub_sub_rt" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main-ig.id
  }
}
resource "aws_route_table_association" "internet_for_pub_sub0" {
  route_table_id = aws_route_table.pub_sub_rt.id
  subnet_id      = element(aws_subnet.public_subnets, 0).id
}
resource "aws_route_table_association" "internet_for_pub_sub1" {
  route_table_id = aws_route_table.pub_sub_rt.id
  subnet_id      = element(aws_subnet.public_subnets, 1).id
}
resource "aws_route_table_association" "internet_for_pub_sub2" {
  route_table_id = aws_route_table.pub_sub_rt.id
  subnet_id      = element(aws_subnet.public_subnets, 2).id
}
# resource "aws_eip" "nat" {
#   vpc = true
# }
# resource "aws_nat_gateway" "nat" {
#   allocation_id = aws_eip.nat[0].id
#   subnet_id     = element(aws_subnet.public_subnets, 0).id
# }
# resource "aws_route_table" "private_subnets" {
#   vpc_id = aws_vpc.main.id
# }
# resource "aws_route_table_association" "private_routes" {
#   count          = length(aws_subnet.private_subnets[*].id)
#   route_table_id = aws_route_table.private_subnets[count.index].id
#   subnet_id      = element(aws_subnet.private_subnets[*].id, count.index)
# }
