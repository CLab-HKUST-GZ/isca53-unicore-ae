module add_int_rca #(
    parameter WIDTH = 8
) (
    input  wire [WIDTH-1:0] Operand_1,
    input  wire [WIDTH-1:0] Operand_2,
    input  wire             Cin,
    output wire [WIDTH-1:0] Sum,
    output wire             Cout
);

    assign {Cout, Sum} = Operand_1 + Operand_2 + Cin;

endmodule