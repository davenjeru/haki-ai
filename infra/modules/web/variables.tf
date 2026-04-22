variable "project_name" {
  type = string
}

variable "price_class" {
  description = "CloudFront price class. PriceClass_100 (US+CA+EU) is the cheapest; PriceClass_200 adds Africa/Middle East/India; PriceClass_All covers everywhere."
  type        = string
  default     = "PriceClass_100"
  validation {
    condition     = contains(["PriceClass_100", "PriceClass_200", "PriceClass_All"], var.price_class)
    error_message = "price_class must be PriceClass_100, PriceClass_200, or PriceClass_All."
  }
}
