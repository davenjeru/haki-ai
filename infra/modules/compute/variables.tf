variable "project_name"      { type = string }
variable "aws_region"        { type = string }
variable "environment"       { type = string }
variable "knowledge_base_id" { type = string }
variable "guardrail_id"      { type = string }
variable "guardrail_version" { type = string }
variable "bedrock_model_id"  { type = string }
variable "chroma_host" {
  type    = string
  default = ""
}
variable "chroma_port" {
  type    = string
  default = "8000"
}
