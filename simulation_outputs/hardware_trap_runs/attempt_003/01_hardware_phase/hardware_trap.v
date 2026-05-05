module hardware_trap (
    input  wire        CLOCK_50,
    output wire [33:0] GPIO_1,
    output wire [15:0] GPIO_0
);

    localparam integer HALF_PERIOD = 625;
    localparam integer PERIOD      = 1250;

    reg [31:0] div_counter = 32'd0;
    reg        out_signal  = 1'b0;

    always @(posedge CLOCK_50) begin
        if (div_counter >= HALF_PERIOD - 1) begin
            div_counter <= 32'd0;
            out_signal  <= ~out_signal;
        end else begin
            div_counter <= div_counter + 32'd1;
        end
    end

    wire [10:0] base_phase =
        (out_signal) ? div_counter[10:0] : (div_counter[10:0] + 11'd625);

    localparam [10:0] TICK_TX01 = 11'd0;
    localparam [10:0] TICK_TX02 = 11'd499;
    localparam [10:0] TICK_TX03 = 11'd673;
    localparam [10:0] TICK_TX04 = 11'd499;
    localparam [10:0] TICK_TX05 = 11'd0;
    localparam [10:0] TICK_TX06 = 11'd499;
    localparam [10:0] TICK_TX07 = 11'd1034;
    localparam [10:0] TICK_TX08 = 11'd1222;
    localparam [10:0] TICK_TX09 = 11'd1034;
    localparam [10:0] TICK_TX10 = 11'd499;
    localparam [10:0] TICK_TX11 = 11'd673;
    localparam [10:0] TICK_TX12 = 11'd1222;
    localparam [10:0] TICK_TX13 = 11'd165;
    localparam [10:0] TICK_TX14 = 11'd1222;
    localparam [10:0] TICK_TX15 = 11'd673;
    localparam [10:0] TICK_TX16 = 11'd499;
    localparam [10:0] TICK_TX17 = 11'd1034;
    localparam [10:0] TICK_TX18 = 11'd1222;
    localparam [10:0] TICK_TX19 = 11'd1034;
    localparam [10:0] TICK_TX20 = 11'd499;
    localparam [10:0] TICK_TX21 = 11'd0;
    localparam [10:0] TICK_TX22 = 11'd499;
    localparam [10:0] TICK_TX23 = 11'd673;
    localparam [10:0] TICK_TX24 = 11'd499;
    localparam [10:0] TICK_TX25 = 11'd0;

    function [10:0] phase_mod;
        input [10:0] a;
        input [10:0] b;
        reg   [11:0] s;
        begin
            s = {1'b0, a} + {1'b0, b};
            if (s >= 12'd1250)
                phase_mod = s - 12'd1250;
            else
                phase_mod = s[10:0];
        end
    endfunction

    wire [24:0] wav_tx;

    wire [10:0] phi01 = phase_mod(base_phase, TICK_TX01);  assign wav_tx[0]  = (phi01 < 11'd625);
    wire [10:0] phi02 = phase_mod(base_phase, TICK_TX02);  assign wav_tx[1]  = (phi02 < 11'd625);
    wire [10:0] phi03 = phase_mod(base_phase, TICK_TX03);  assign wav_tx[2]  = (phi03 < 11'd625);
    wire [10:0] phi04 = phase_mod(base_phase, TICK_TX04);  assign wav_tx[3]  = (phi04 < 11'd625);
    wire [10:0] phi05 = phase_mod(base_phase, TICK_TX05);  assign wav_tx[4]  = (phi05 < 11'd625);
    wire [10:0] phi06 = phase_mod(base_phase, TICK_TX06);  assign wav_tx[5]  = (phi06 < 11'd625);
    wire [10:0] phi07 = phase_mod(base_phase, TICK_TX07);  assign wav_tx[6]  = (phi07 < 11'd625);
    wire [10:0] phi08 = phase_mod(base_phase, TICK_TX08);  assign wav_tx[7]  = (phi08 < 11'd625);
    wire [10:0] phi09 = phase_mod(base_phase, TICK_TX09);  assign wav_tx[8]  = (phi09 < 11'd625);
    wire [10:0] phi10 = phase_mod(base_phase, TICK_TX10);  assign wav_tx[9]  = (phi10 < 11'd625);
    wire [10:0] phi11 = phase_mod(base_phase, TICK_TX11);  assign wav_tx[10] = (phi11 < 11'd625);
    wire [10:0] phi12 = phase_mod(base_phase, TICK_TX12);  assign wav_tx[11] = (phi12 < 11'd625);
    wire [10:0] phi13 = phase_mod(base_phase, TICK_TX13);  assign wav_tx[12] = (phi13 < 11'd625);
    wire [10:0] phi14 = phase_mod(base_phase, TICK_TX14);  assign wav_tx[13] = (phi14 < 11'd625);
    wire [10:0] phi15 = phase_mod(base_phase, TICK_TX15);  assign wav_tx[14] = (phi15 < 11'd625);
    wire [10:0] phi16 = phase_mod(base_phase, TICK_TX16);  assign wav_tx[15] = (phi16 < 11'd625);
    wire [10:0] phi17 = phase_mod(base_phase, TICK_TX17);  assign wav_tx[16] = (phi17 < 11'd625);
    wire [10:0] phi18 = phase_mod(base_phase, TICK_TX18);  assign wav_tx[17] = (phi18 < 11'd625);
    wire [10:0] phi19 = phase_mod(base_phase, TICK_TX19);  assign wav_tx[18] = (phi19 < 11'd625);
    wire [10:0] phi20 = phase_mod(base_phase, TICK_TX20);  assign wav_tx[19] = (phi20 < 11'd625);
    wire [10:0] phi21 = phase_mod(base_phase, TICK_TX21);  assign wav_tx[20] = (phi21 < 11'd625);
    wire [10:0] phi22 = phase_mod(base_phase, TICK_TX22);  assign wav_tx[21] = (phi22 < 11'd625);
    wire [10:0] phi23 = phase_mod(base_phase, TICK_TX23);  assign wav_tx[22] = (phi23 < 11'd625);
    wire [10:0] phi24 = phase_mod(base_phase, TICK_TX24);  assign wav_tx[23] = (phi24 < 11'd625);
    wire [10:0] phi25 = phase_mod(base_phase, TICK_TX25);  assign wav_tx[24] = (phi25 < 11'd625);

    genvar i;
    generate
        for (i = 0; i < 17; i = i + 1) begin : trans_1_to_17
            assign GPIO_1[2*i]     =  wav_tx[i];
            assign GPIO_1[2*i + 1] = ~wav_tx[i];
        end

        for (i = 0; i < 8; i = i + 1) begin : trans_18_to_25
            assign GPIO_0[2*i]     =  wav_tx[17 + i];
            assign GPIO_0[2*i + 1] = ~wav_tx[17 + i];
        end
    endgenerate

endmodule
