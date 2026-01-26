from AlgorithmImports import *


class SchwabFeeModel(FeeModel):
    def get_order_fee(self, parameters: OrderFeeParameters) -> OrderFee:
        commission_per_contract = 2.6
        fee_per_contract = 2.01
        return OrderFee(CashAmount(0.5, "USD"))
