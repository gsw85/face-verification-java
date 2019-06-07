package com.skymindglobal.faceverification.common;

import okhttp3.*;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

public class SkilClient
{
    private final String username;
    private final String password;
    private String token;
    private String host;
    private String port;

    public SkilClient(String username, String password, String host, String port)
    {
        this.username = username;
        this.password = password;
        this.host = host;
        this.port = port;
        this.login();
    }

    String extractHostAndPort(String endpoint)
    {
        // http://foo:9008/bar/baz
        //^---------------^ Return this part.
        int thirdSlash = endpoint.indexOf("/", "http://".length());
        return endpoint.substring(0, thirdSlash);
    }

    String extractToken(String tokenJson)
    {
        // { "token": "......"}
        int colon = tokenJson.indexOf(":");
        int firstQuote = tokenJson.indexOf("\"", colon);
        int lastQuote = tokenJson.indexOf("\"", firstQuote + 1);

        return tokenJson.substring(firstQuote + 1, lastQuote);
    }

    List<Integer> extractResults(String jsonRes)
    {
        int resultsIndex = jsonRes.indexOf("\"results\":", 0);
        int firstBracket = jsonRes.indexOf("[", resultsIndex);
        int secondBracket = jsonRes.indexOf("]", firstBracket);

        String[] results = jsonRes.substring(firstBracket + 1, secondBracket).split(",");
        int[] res = new int[results.length];
        return Arrays.stream(results).map(String::trim).map(Integer::valueOf).collect(Collectors.toList());
    }

    List<Float> extractProbs(String jsonRes) {
        int probsIndex = jsonRes.indexOf("\"probabilities\":", 0);
        int probsFirstBracket = jsonRes.indexOf("[", probsIndex);
        int probsSecondBracket = jsonRes.indexOf("]", probsFirstBracket);

        String[] probs = jsonRes.substring(probsFirstBracket + 1, probsSecondBracket).split(",");

        return Arrays.stream(probs).map(String::trim).map(Float::valueOf).collect(Collectors.toList());
    }


    public String classify(String endpoint, INDArray input) throws NullPointerException
    {
        OkHttpClient client = new OkHttpClient();

        MediaType mediaType = MediaType.parse("application/json; charset=utf-8");
        String reqUUID = UUID.randomUUID().toString();

        String jsonReq =
                "{\"prediction\": {" +
                            "\"inputs\":" +
                                "[" +
                                    "{"+
                                        "\"array\":\""+ input.data().toString() + "\","+
                                        "\"shape\":\"[3,224,224]\","+
                                        "\"ordering\":\"c\","+
                                        "\"data\":\""+ input.data().toString() +"\""+
//                                        "\"dataType\":\"FLOAT16\"" +
                                    "}" +
                                "]," +
                            "\"id\":\"" + reqUUID + "\"," +
                            "\"needsPreProcessing\":\"false\"" +
                        "}";

        RequestBody requestBody = RequestBody.create(mediaType, jsonReq);

        Request request = new Request.Builder()
                .header(
                        "Authorization",
                        "Bearer " + this.token)//"JWT" + " " +  this.token)
                .url(endpoint + "multipredict")
                .post(requestBody)
                .build();

        try
        {
            Response response = client.newCall(request).execute();
            if(response.isSuccessful())
            {
                System.out.println("Request success!");
                return response.body().string();
            }
            else
            {
                System.out.println("Request failed.");
                System.out.println(response.message());
            }
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        return null;
    }

    private void login()
    {
        OkHttpClient client = new OkHttpClient();
        MediaType mediaType = MediaType.parse("application/json");

        try
        {
            RequestBody body = RequestBody.create(mediaType, "{\"userId\":\"" + this.username + "\",\"password\":\"" + this.password + "\"}");
            Request request = new Request.Builder().url("http://" + this.host + ":" + this.port + "/login").post(body).build();
            Response response = client.newCall(request).execute();
            if(response.isSuccessful())
            {
                String responseBody = response.body().string();
                this.token = extractToken(responseBody);
            }
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }
    }
}
